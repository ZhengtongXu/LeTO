from collections import OrderedDict
import textwrap
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import robomimic.utils.tensor_utils as TensorUtils
from robomimic.models.base_nets import Module
from robomimic.models.obs_nets import ObservationGroupEncoder
from robomimic.models.obs_nets import  ObservationDecoder
from robomimic.models.base_nets import Sequential, MLP
from robomimic.algo.bc import BC
import robomimic.models.base_nets as BaseNets
import robomimic.models.policy_nets as PolicyNets
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from copy import deepcopy
from collections import OrderedDict

from qpth.qp import QPFunction, QPSolvers
from torch.autograd import Variable
from torch.nn.parameter import Parameter as Parame


from robomimic.algo import register_algo_factory_func, PolicyAlgo


class LeTO_RNN_Base(Module):
    """
    A wrapper class for a multi-step RNN and a per-step network.
    """
    def __init__(
        self,
        input_dim,
        rnn_hidden_dim,
        rnn_num_layers,
        rnn_type="LSTM",  # [LSTM, GRU]
        rnn_kwargs=None,
        per_step_net=None,
        pred_horizon = 6,
        single_ac_dim = 7,
        device = 'cuda:0',
        smooth_weight = 1,
    ):
        """
        Args:
            input_dim (int): dimension of inputs

            rnn_hidden_dim (int): RNN hidden dimension

            rnn_num_layers (int): number of RNN layers

            rnn_type (str): [LSTM, GRU]

            rnn_kwargs (dict): kwargs for the torch.nn.LSTM / GRU

            per_step_net: a network that runs per time step on top of the RNN output
        """
        super(LeTO_RNN_Base, self).__init__()
        self.per_step_net = per_step_net
        if per_step_net is not None:
            assert isinstance(per_step_net, Module), "LeTO_RNN_Base: per_step_net is not instance of Module"

        assert rnn_type in ["LSTM", "GRU"]
        rnn_cls = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        rnn_kwargs = rnn_kwargs if rnn_kwargs is not None else {}
        rnn_is_bidirectional = rnn_kwargs.get("bidirectional", False)

        self.nets = rnn_cls(
            input_size=input_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_num_layers,
            batch_first=True,
            **rnn_kwargs,
        )
        self.device = device
        self._hidden_dim = rnn_hidden_dim
        self._num_layers = rnn_num_layers
        self._rnn_type = rnn_type
        self._num_directions = int(rnn_is_bidirectional) + 1 # 2 if bidirectional, 1 otherwise
        self.smooth_weight = smooth_weight
        self.single_ac_dim = single_ac_dim
        act_dim = single_ac_dim
        n_pred = pred_horizon
        self.L = Parame(torch.tril(torch.rand(act_dim*n_pred, act_dim*n_pred).to(self.device)))

        self.eps = 1e-4

        self.ouput_dim = act_dim*n_pred

        sub_diag = torch.eye(act_dim, requires_grad = False)
        sub_diff = torch.cat((sub_diag, -sub_diag), 1)

        A_sat = torch.cat((sub_diff,torch.zeros(act_dim,act_dim*n_pred - act_dim*2,requires_grad = False)),1)

        for i in range(1,n_pred-1):
            row = torch.cat((torch.zeros(act_dim,act_dim*i,requires_grad = False),sub_diff),1)
            row = torch.cat((row,torch.zeros(act_dim,act_dim*n_pred - act_dim*(2+i),requires_grad = False)),1)
            A_sat = torch.cat((A_sat,row),0)

        A_vel = torch.eye(act_dim*n_pred, requires_grad = False)

        if act_dim == 7:
            for i in range (n_pred-1):
                A_sat = A_sat[torch.arange(A_sat.size(0))!= (n_pred-i-1)*act_dim - 1]
            for i in range (n_pred):
                A_vel = A_vel[torch.arange(A_vel.size(0))!= (n_pred-i)*act_dim - 1]
            self.b_sat_single = torch.tensor([0.1,0.1,0.1,0.1,0.1,0.1],requires_grad = False).resize(act_dim-1,1)
            self.b_vel_single = torch.tensor([1,1,1,1,1,1],requires_grad = False).resize(act_dim-1,1)
        else:
            assert (act_dim == 7)
        
        self.A_sat_single = A_sat
        self.A_sat = torch.cat((A_sat,-A_sat),0)

        self.A_vel = torch.cat((A_vel,-A_vel),0)

        self.b_sat  = self.b_sat_single
        for i in range(2*(n_pred-1)-1):
            self.b_sat = torch.cat((self.b_sat,self.b_sat_single),0)

        b_vel_dual = self.b_vel_single
        for i in range(n_pred-1):
            b_vel_dual = torch.cat((b_vel_dual,self.b_vel_single),0)
        self.b_vel = torch.cat((b_vel_dual,b_vel_dual),0)

        if act_dim == 7:
            A_single = torch.eye(act_dim-1,requires_grad = False)
            A_single_zero = torch.cat((A_single,torch.zeros(act_dim-1,act_dim*n_pred - act_dim+1,requires_grad = False)),1)
            self.A_past_action = torch.cat((A_single_zero,-A_single_zero),0)
        else:
            assert (act_dim == 7)

    @property
    def rnn_type(self):
        return self._rnn_type

    def get_rnn_init_state(self, batch_size, device):
        """
        Get a default RNN state (zeros)
        Args:
            batch_size (int): batch size dimension

            device: device the hidden state should be sent to.

        Returns:
            hidden_state (torch.Tensor or tuple): returns hidden state tensor or tuple of hidden state tensors
                depending on the RNN type
        """
        h_0 = torch.zeros(self._num_layers * self._num_directions, batch_size, self._hidden_dim).to(device)
        if self._rnn_type == "LSTM":
            c_0 = torch.zeros(self._num_layers * self._num_directions, batch_size, self._hidden_dim).to(device)
            return h_0, c_0
        else:
            return h_0

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # infer time dimension from input shape and add to per_step_net output shape
        if self.per_step_net is not None:
            out = self.per_step_net.output_shape(input_shape[1:])
            if isinstance(out, dict):
                out = {k: [input_shape[0]] + out[k] for k in out}
            else:
                out = [input_shape[0]] + out
        else:
            out = [input_shape[0], self._num_layers * self._hidden_dim]
        return out

    def forward(self, inputs, rnn_init_state=None, return_state=False):
        """
        Forward a sequence of inputs through the RNN and the per-step network.

        Args:
            inputs (torch.Tensor): tensor input of shape [B, T, D], where D is the RNN input size

            rnn_init_state: rnn hidden state, initialize to zero state if set to None

            return_state (bool): whether to return hidden state

        Returns:
            outputs: outputs of the per_step_net

            rnn_state: return rnn state at the end if return_state is set to True
        """
        assert inputs.ndimension() == 3  # [B, T, D]
        batch_size, seq_length, inp_dim = inputs.shape
        if rnn_init_state is None:
            rnn_init_state = self.get_rnn_init_state(batch_size, device=inputs.device)

        outputs, rnn_state = self.nets(inputs, rnn_init_state)
        if self.per_step_net is not None:
            outputs = TensorUtils.time_distributed(outputs, self.per_step_net)

        mean = outputs['action']

        batch_size, horizon, act_dim = mean.shape
        n_batch = batch_size*horizon
        x = mean.view(n_batch, act_dim)

        L = self.L
        Q = L.mm(L.t()) + self.eps*(torch.eye(self.ouput_dim,requires_grad = False)).to(inputs.device) + self.smooth_weight*(self.A_sat_single.t().mm(self.A_sat_single)).to(inputs.device)
        Q = Q.unsqueeze(0).expand(n_batch, self.ouput_dim, self.ouput_dim)

        e = Variable(torch.Tensor())

        # map to [-1,1]
        inputs = -torch.tanh(x)

        A_sat_batch = self.A_sat.unsqueeze(0).expand(n_batch, self.A_sat.shape[0], self.A_sat.shape[1]).to(inputs.device)
        b_sat_batch = self.b_sat.unsqueeze(0).expand(n_batch, self.b_sat.shape[0], 1).to(inputs.device).reshape(n_batch, self.b_sat.shape[0])
        A_vel_batch = self.A_vel.unsqueeze(0).expand(n_batch, self.A_vel.shape[0], self.A_vel.shape[1]).to(inputs.device)
        b_vel_batch = self.b_vel.unsqueeze(0).expand(n_batch, self.b_vel.shape[0], 1).to(inputs.device).reshape(n_batch, self.b_vel.shape[0])

        G = torch.cat((A_sat_batch,A_vel_batch),1)
        h = torch.cat((b_sat_batch,b_vel_batch),1)
        
        x = QPFunction(verbose=-1)(
            Q.double(), inputs.double(), G.double(), h.double(), e, e)
        x = x.float()
        mean = x.view(batch_size, horizon, act_dim)
        outputs['action'] = mean
   

        if return_state:
            return outputs, rnn_state
        else:
            return outputs

    def forward_with_past_action(self, inputs, rnn_init_state=None, return_state=False,past_action = None):

        assert inputs.ndimension() == 3  # [B, T, D]
        batch_size, seq_length, inp_dim = inputs.shape
        if rnn_init_state is None:
            rnn_init_state = self.get_rnn_init_state(batch_size, device=inputs.device)

        outputs, rnn_state = self.nets(inputs, rnn_init_state)

        if self.per_step_net is not None:
            outputs = TensorUtils.time_distributed(outputs, self.per_step_net)

        mean = outputs['action']

        batch_size, horizon, act_dim = mean.shape
        n_batch = batch_size*horizon
        x = mean.view(n_batch, act_dim)


        L = self.L
        Q = L.mm(L.t()) + self.eps*(torch.eye(self.ouput_dim,requires_grad = False)).to(inputs.device) + self.smooth_weight*(self.A_sat_single.t().mm(self.A_sat_single)).to(inputs.device)
        Q = Q.unsqueeze(0).expand(n_batch, self.ouput_dim, self.ouput_dim)

        e = Variable(torch.Tensor())

        # map to [-1,1]
        inputs = -torch.tanh(x)

        
        A_sat_batch = self.A_sat.unsqueeze(0).expand(n_batch, self.A_sat.shape[0], self.A_sat.shape[1]).to(inputs.device)
        b_sat_batch = self.b_sat.unsqueeze(0).expand(n_batch, self.b_sat.shape[0], 1).to(inputs.device).reshape(n_batch, self.b_sat.shape[0])

        A_vel_batch = self.A_vel.unsqueeze(0).expand(n_batch, self.A_vel.shape[0], self.A_vel.shape[1]).to(inputs.device)
        b_vel_batch = self.b_vel.unsqueeze(0).expand(n_batch, self.b_vel.shape[0], 1).to(inputs.device).reshape(n_batch, self.b_vel.shape[0])

        A_past_action_batch = self.A_past_action.unsqueeze(0).expand(n_batch, self.A_past_action.shape[0], self.ouput_dim).to(inputs.device)

        if self.single_ac_dim == 7:
            b_past_action_batch_upper = past_action[:,:,0:6].reshape(n_batch, 6) + 0.1*torch.ones(n_batch, 6,requires_grad = False).to(inputs.device) 
            b_past_action_batch_lower = -1*past_action[:,:,0:6].reshape(n_batch, 6) + 0.1*torch.ones(n_batch, 6,requires_grad = False).to(inputs.device)


        b_past_action_batch = torch.cat((b_past_action_batch_upper,b_past_action_batch_lower),1)   


        G = torch.cat((A_sat_batch,A_vel_batch,A_past_action_batch),1)
        h = torch.cat((b_sat_batch,b_vel_batch,b_past_action_batch),1)

        x = QPFunction(verbose=-1)(
            Q.double(), inputs.double(), G.double(), h.double(), e, e)
        x = x.float()

        mean = x.view(batch_size, horizon, act_dim)
        outputs['action'] = mean


        if return_state:
            return outputs, rnn_state
        else:
            return outputs

    def forward_step(self, inputs, rnn_state):
        """
        Forward a single step input through the RNN and per-step network, and return the new hidden state.
        Args:
            inputs (torch.Tensor): tensor input of shape [B, D], where D is the RNN input size

            rnn_state: rnn hidden state, initialize to zero state if set to None

        Returns:
            outputs: outputs of the per_step_net

            rnn_state: return the new rnn state
        """
        assert inputs.ndimension() == 2
        inputs = TensorUtils.to_sequence(inputs)
        outputs, rnn_state = self.forward(
            inputs,
            rnn_init_state=rnn_state,
            return_state=True,
        )
        return outputs[:, 0], rnn_state




class LeTO_RNN_MLP(Module):
    """
    A wrapper class for a multi-step RNN and a per-step MLP and a decoder.

    Structure: [encoder -> rnn -> mlp -> decoder]

    All temporal inputs are processed by a shared @ObservationGroupEncoder,
    followed by an RNN, and then a per-step multi-output MLP. 
    """
    def __init__(
        self,
        input_obs_group_shapes,
        output_shapes,
        mlp_layer_dims,
        rnn_hidden_dim,
        rnn_num_layers,
        rnn_type="LSTM",  # [LSTM, GRU]
        rnn_kwargs=None,
        mlp_activation=nn.ReLU,
        mlp_layer_func=nn.Linear,
        per_step=True,
        encoder_kwargs=None,
        pred_horizon = 6,
        single_ac_dim = 7,
        device = 'cuda:0',
        smooth_weight = 1,
    ):
        """
        Args:
            input_obs_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.

            output_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for outputs.

            rnn_hidden_dim (int): RNN hidden dimension

            rnn_num_layers (int): number of RNN layers

            rnn_type (str): [LSTM, GRU]

            rnn_kwargs (dict): kwargs for the rnn model

            per_step (bool): if True, apply the MLP and observation decoder into @output_shapes
                at every step of the RNN. Otherwise, apply them to the final hidden state of the 
                RNN.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        super(LeTO_RNN_MLP, self).__init__()
        assert isinstance(input_obs_group_shapes, OrderedDict)
        assert np.all([isinstance(input_obs_group_shapes[k], OrderedDict) for k in input_obs_group_shapes])
        assert isinstance(output_shapes, OrderedDict)
        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_shapes = output_shapes
        self.per_step = per_step

        self.nets = nn.ModuleDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )

        # flat encoder output dimension
        rnn_input_dim = self.nets["encoder"].output_shape()[0]

        # bidirectional RNNs mean that the output of RNN will be twice the hidden dimension
        rnn_is_bidirectional = rnn_kwargs.get("bidirectional", False)
        num_directions = int(rnn_is_bidirectional) + 1 # 2 if bidirectional, 1 otherwise
        rnn_output_dim = num_directions * rnn_hidden_dim

        per_step_net = None
        self._has_mlp = (len(mlp_layer_dims) > 0)
        if self._has_mlp:
            self.nets["mlp"] = MLP(
                input_dim=rnn_output_dim,
                output_dim=mlp_layer_dims[-1],
                layer_dims=mlp_layer_dims[:-1],
                output_activation=mlp_activation,
                layer_func=mlp_layer_func
            )
            self.nets["decoder"] = ObservationDecoder(
                decode_shapes=self.output_shapes,
                input_feat_dim=mlp_layer_dims[-1],
            )
            if self.per_step:
                per_step_net = Sequential(self.nets["mlp"], self.nets["decoder"])
        else:
            self.nets["decoder"] = ObservationDecoder(
                decode_shapes=self.output_shapes,
                input_feat_dim=rnn_output_dim,
            )
            if self.per_step:
                per_step_net = self.nets["decoder"]

        # core network
        self.nets["rnn"] = LeTO_RNN_Base(
            input_dim=rnn_input_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            rnn_type=rnn_type,
            per_step_net=per_step_net,
            rnn_kwargs=rnn_kwargs,
            pred_horizon = pred_horizon,
            single_ac_dim = single_ac_dim,
            device = device,
            smooth_weight = smooth_weight,
        )

    def get_rnn_init_state(self, batch_size, device):
        """
        Get a default RNN state (zeros)

        Args:
            batch_size (int): batch size dimension

            device: device the hidden state should be sent to.

        Returns:
            hidden_state (torch.Tensor or tuple): returns hidden state tensor or tuple of hidden state tensors
                depending on the RNN type
        """
        return self.nets["rnn"].get_rnn_init_state(batch_size, device=device)

    def output_shape(self, input_shape):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.

        Args:
            input_shape (dict): dictionary of dictionaries, where each top-level key
                corresponds to an observation group, and the low-level dictionaries
                specify the shape for each modality in an observation dictionary
        """

        # infers temporal dimension from input shape
        obs_group = list(self.input_obs_group_shapes.keys())[0]
        mod = list(self.input_obs_group_shapes[obs_group].keys())[0]
        T = input_shape[obs_group][mod][0]
        TensorUtils.assert_size_at_dim(input_shape, size=T, dim=0, 
                msg="LeTO_RNN_MLP: input_shape inconsistent in temporal dimension")
        # returns a dictionary instead of list since outputs are dictionaries
        return { k : [T] + list(self.output_shapes[k]) for k in self.output_shapes }

    def forward(self, rnn_init_state=None, return_state=False, **inputs):
        """
        Args:
            inputs (dict): a dictionary of dictionaries with one dictionary per
                observation group. Each observation group's dictionary should map
                modality to torch.Tensor batches. Should be consistent with
                @self.input_obs_group_shapes. First two leading dimensions should
                be batch and time [B, T, ...] for each tensor.

            rnn_init_state: rnn hidden state, initialize to zero state if set to None

            return_state (bool): whether to return hidden state

        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes. Leading dimensions will be batch and time [B, T, ...]
                for each tensor.

            rnn_state (torch.Tensor or tuple): return the new rnn state (if @return_state)
        """


        for obs_group in self.input_obs_group_shapes:
            for k in self.input_obs_group_shapes[obs_group]:
                # first two dimensions should be [B, T] for inputs
                assert inputs[obs_group][k].ndim - 2 == len(self.input_obs_group_shapes[obs_group][k])

        # use encoder to extract flat rnn inputs
        rnn_inputs = TensorUtils.time_distributed(inputs, self.nets["encoder"], inputs_as_kwargs=True)
        assert rnn_inputs.ndim == 3  # [B, T, D]
        if self.per_step:
            if 'past_action' not in inputs['obs'].keys():
                return self.nets["rnn"].forward(inputs=rnn_inputs, rnn_init_state=rnn_init_state, return_state=return_state)
            else:
                return self.nets["rnn"].forward_with_past_action(inputs=rnn_inputs, rnn_init_state=rnn_init_state, return_state=return_state, past_action=inputs['obs']['past_action'])
        # apply MLP + decoder to last RNN output
        outputs = self.nets["rnn"].forward(inputs=rnn_inputs, rnn_init_state=rnn_init_state, return_state=return_state)
        if return_state:
            outputs, rnn_state = outputs

        assert outputs.ndim == 3 # [B, T, D]
        if self._has_mlp:
            outputs = self.nets["decoder"](self.nets["mlp"](outputs[:, -1]))
        else:
            outputs = self.nets["decoder"](outputs[:, -1])

        if return_state:
            return outputs, rnn_state
        return outputs

    def forward_step(self, rnn_state, **inputs):
        """
        Unroll network over a single timestep.

        Args:
            inputs (dict): expects same modalities as @self.input_shapes, with
                additional batch dimension (but NOT time), since this is a 
                single time step.

            rnn_state (torch.Tensor): rnn hidden state

        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes. Does not contain time dimension.

            rnn_state: return the new rnn state
        """
        # ensure that the only extra dimension is batch dim, not temporal dim 
        assert np.all([inputs[k].ndim - 1 == len(self.input_shapes[k]) for k in self.input_shapes])

        inputs = TensorUtils.to_sequence(inputs)
        outputs, rnn_state = self.forward(
            inputs, 
            rnn_init_state=rnn_state,
            return_state=True,
        )
        if self.per_step:
            # if outputs are not per-step, the time dimension is already reduced
            outputs = outputs[:, 0]
        return outputs, rnn_state

    def _to_string(self):
        """
        Subclasses should override this method to print out info about network / policy.
        """
        return ''

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        msg += textwrap.indent("\n" + self._to_string(), indent)
        msg += textwrap.indent("\n\nencoder={}".format(self.nets["encoder"]), indent)
        msg += textwrap.indent("\n\nrnn={}".format(self.nets["rnn"]), indent)
        msg = header + '(' + msg + '\n)'
        return msg



class LeTORNNActorNetwork(LeTO_RNN_MLP):
    """
    An RNN policy network that predicts actions from observations.
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        mlp_layer_dims,
        rnn_hidden_dim,
        rnn_num_layers,
        rnn_type="LSTM",  # [LSTM, GRU]
        rnn_kwargs=None,
        goal_shapes=None,
        encoder_kwargs=None,
        pred_horizon = 6,
        single_ac_dim = 7,
        device = 'cuda:0',
        smooth_weight = 1,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes.

            rnn_hidden_dim (int): RNN hidden dimension

            rnn_num_layers (int): number of RNN layers

            rnn_type (str): [LSTM, GRU]

            rnn_kwargs (dict): kwargs for the torch.nn.LSTM / GRU

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        self.ac_dim = ac_dim
        self.device = device
        assert isinstance(obs_shapes, OrderedDict)
        self.obs_shapes = obs_shapes

        # set up different observation groups for @LeTO_RNN_MLP
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)

        self._is_goal_conditioned = False
        if goal_shapes is not None and len(goal_shapes) > 0:
            assert isinstance(goal_shapes, OrderedDict)
            self._is_goal_conditioned = True
            self.goal_shapes = OrderedDict(goal_shapes)
            observation_group_shapes["goal"] = OrderedDict(self.goal_shapes)
        else:
            self.goal_shapes = OrderedDict()

        output_shapes = self._get_output_shapes()
        super(LeTORNNActorNetwork, self).__init__(
            input_obs_group_shapes=observation_group_shapes,
            output_shapes=output_shapes,
            mlp_layer_dims=mlp_layer_dims,
            mlp_activation=nn.ReLU,
            mlp_layer_func=nn.Linear,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            rnn_type=rnn_type,
            rnn_kwargs=rnn_kwargs,
            per_step=True,
            encoder_kwargs=encoder_kwargs,
            pred_horizon = pred_horizon,
            single_ac_dim = single_ac_dim,
            device = device,
            smooth_weight   = smooth_weight,
        )

    def _get_output_shapes(self):
        """
        Allow subclasses to re-define outputs from @LeTO_RNN_MLP, since we won't
        always directly predict actions, but may instead predict the parameters
        of a action distribution.
        """
        return OrderedDict(action=(self.ac_dim,))

    def output_shape(self, input_shape):
        # note: @input_shape should be dictionary (key: mod)
        # infers temporal dimension from input shape
        mod = list(self.obs_shapes.keys())[0]
        T = input_shape[mod][0]
        TensorUtils.assert_size_at_dim(input_shape, size=T, dim=0, 
                msg="LeTORNNActorNetwork: input_shape inconsistent in temporal dimension")
        return [T, self.ac_dim]

    def forward(self, obs_dict, goal_dict=None, rnn_init_state=None, return_state=False):
        """
        Forward a sequence of inputs through the RNN and the per-step network.

        Args:
            obs_dict (dict): batch of observations - each tensor in the dictionary
                should have leading dimensions batch and time [B, T, ...]
            goal_dict (dict): if not None, batch of goal observations
            rnn_init_state: rnn hidden state, initialize to zero state if set to None
            return_state (bool): whether to return hidden state

        Returns:
            actions (torch.Tensor): predicted action sequence
            rnn_state: return rnn state at the end if return_state is set to True
        """
        if self._is_goal_conditioned:
            assert goal_dict is not None
            # repeat the goal observation in time to match dimension with obs_dict
            mod = list(obs_dict.keys())[0]
            goal_dict = TensorUtils.unsqueeze_expand_at(goal_dict, size=obs_dict[mod].shape[1], dim=1)

        outputs = super(LeTORNNActorNetwork, self).forward(
            obs=obs_dict, goal=goal_dict, rnn_init_state=rnn_init_state, return_state=return_state) 
        
        if return_state:
            actions, state = outputs
        else:
            actions = outputs
            state = None
        # apply tanh squashing to ensure actions are in [-1, 1]
        # actions = torch.tanh(actions["action"])
        actions = actions["action"]

        if return_state:
            return actions, state
        else:
            return actions

    def forward_step(self, obs_dict, goal_dict=None, rnn_state=None):
        """
        Unroll RNN over single timestep to get actions.

        Args:
            obs_dict (dict): batch of observations. Should not contain
                time dimension.
            goal_dict (dict): if not None, batch of goal observations
            rnn_state: rnn hidden state, initialize to zero state if set to None

        Returns:
            actions (torch.Tensor): batch of actions - does not contain time dimension
            state: updated rnn state
        """
        obs_dict = TensorUtils.to_sequence(obs_dict)
        action, state = self.forward(
            obs_dict, goal_dict, rnn_init_state=rnn_state, return_state=True)
        return action[:, 0], state

    def _to_string(self):
        """Info to pretty print."""
        return "action_dim={}".format(self.ac_dim)
    

class LeTO(BC):
    """
    BC training with an RNN policy.
    """
    def __init__(
        self,
        algo_config,
        obs_config,
        global_config,
        obs_key_shapes,
        ac_dim,
        device,
        single_ac_dim,
        pred_horizon,
        samp_horizon,
        smooth_weight,
    ):
        """
        Args:
            algo_config (Config object): instance of Config corresponding to the algo section
                of the config

            obs_config (Config object): instance of Config corresponding to the observation
                section of the config

            global_config (Config object): global training config

            obs_key_shapes (OrderedDict): dictionary that maps observation keys to shapes

            ac_dim (int): dimension of action space

            device (torch.Device): where the algo should live (i.e. cpu, gpu)
        """
        self.optim_params = deepcopy(algo_config.optim_params)
        self.algo_config = algo_config
        self.obs_config = obs_config
        self.global_config = global_config

        self.ac_dim = ac_dim
        self.device = device
        self.obs_key_shapes = obs_key_shapes

        self.single_ac_dim = single_ac_dim
        self.pred_horizon = pred_horizon
        self.samp_horizon = samp_horizon
        self.smooth_weight = smooth_weight

        self.nets = nn.ModuleDict()
        self._create_shapes(obs_config.modalities, obs_key_shapes)
        self._create_networks()
        self._create_optimizers()
        assert isinstance(self.nets, nn.ModuleDict)

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.nets = nn.ModuleDict()
        self.nets["policy"] = LeTORNNActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
            pred_horizon = self.pred_horizon,
            single_ac_dim = self.single_ac_dim,
            device=self.device,
            smooth_weight = self.smooth_weight,
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.samp_horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)



    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """

        Tsa = self.samp_horizon
        Tpred = self.pred_horizon
        Dsa = self.single_ac_dim

        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, 0:Tsa, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        n_batch, time_step, act_dim = batch["actions"].shape


        temp = torch.zeros([n_batch,Tsa,Tpred*Dsa])

        for i in range(Tsa):
            temp[:,i,:] = batch["actions"][:,i:i+Tpred,:].reshape(n_batch,Tpred*Dsa)

        input_batch["actions"] = temp

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def get_action(self, obs_dict, goal_dict=None):
        """
        Get policy action outputs.

        Args:
            obs_dict (dict): current observation
            goal_dict (dict): (optional) goal

        Returns:
            action (torch.Tensor): action tensor
        """
        assert not self.nets.training

        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.nets["policy"].get_rnn_init_state(batch_size=batch_size, device=self.device)
            if self._rnn_is_open_loop:
                # remember the initial observation, and use it instead of the current observation
                # for open-loop action sequence prediction
                self._open_loop_obs = TensorUtils.clone(TensorUtils.detach(obs_dict))

        obs_to_use = obs_dict

        if self._rnn_is_open_loop:
            # replace current obs with last recorded obs
            obs_to_use = self._open_loop_obs
        self._rnn_counter += 1
        action, self._rnn_hidden_state = self.nets["policy"].forward_step(
            obs_to_use, goal_dict=goal_dict, rnn_state=self._rnn_hidden_state)
        return action

    def reset(self):
        """
        Reset algo state to prepare for environment rollouts.
        """
        self._rnn_hidden_state = None
        self._rnn_counter = 0