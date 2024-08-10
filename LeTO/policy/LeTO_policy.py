from typing import Dict
import torch
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply

from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
from diffusion_policy.common.robomimic_config_util import get_robomimic_config

from LeTO.model.LeTO import LeTO
from LeTO.model.normalizer import LinearNormalizer
class LetoPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            algo_name='bc_rnn',
            obs_type='image',
            task_name='square',
            dataset_type='ph',
            crop_shape=(76,76),
            pred_steps=1,
            n_action_steps=1,
            samp_steps=1,
            smooth_weight=1,
            constraints=0.1,
        ):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name=algo_name,
            hdf5_type=obs_type,
            task_name=task_name,
            dataset_type=dataset_type)

        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        model = LeTO(
                algo_config=config.algo,
                obs_config=config.observation,
                global_config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim*pred_steps,
                device='cpu',
                single_ac_dim=action_dim,
                pred_horizon=pred_steps,
                samp_horizon=samp_steps,
                smooth_weight=smooth_weight,
                constraints = constraints,
            )

        self.model = model
        self.nets = model.nets
        self.normalizer = LinearNormalizer()
        self.config = config
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim
        self.pred_steps = pred_steps
    def to(self,*args,**kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.model.device = device
        super().to(*args,**kwargs)
    
    # =========== inference =============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs_dict = self.normalizer(obs_dict)
        leto_obs_dict = dict_apply(nobs_dict, lambda x: x[:,0,...])
        naction = self.model.get_action(leto_obs_dict)
        action = self.normalizer['action'].unnormalize(naction)
        # (B, Da)
        result = {
            'action': action[:,0:self.n_action_steps*self.action_dim].reshape(-1, self.n_action_steps, self.action_dim) # (B, n_action_steps, Da)
        }
        return result


    def predict_action_eval(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs_dict = self.normalizer(obs_dict)
        leto_obs_dict = dict_apply(nobs_dict, lambda x: x[:,0,...])
        naction = self.model.get_action(leto_obs_dict)
        action = self.normalizer['action'].unnormalize(naction)
        # (B, Da)
        result = {
            'action': action[:,None,0:self.action_dim] # (B, 1, Da)
        }
        return result


    def reset(self):
        self.model.reset()

    # =========== training ==============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def train_on_batch(self, batch, epoch, validate=False):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        leto_batch = {
            'obs': nobs,
            'actions': nactions
        }
        input_batch = self.model.process_batch_for_training(
            leto_batch)
        info = self.model.train_on_batch(
            batch=input_batch, epoch=epoch, validate=validate)
        # keys: losses, predictions
        return info
    
    def on_epoch_end(self, epoch):
        self.model.on_epoch_end(epoch)

    def get_optimizer(self):
        return self.model.optimizers['policy']




