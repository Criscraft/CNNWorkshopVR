import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import enum
import random

from Scripts.TrackingModules import TrackerModuleProvider

tm = TrackerModuleProvider()

class WeightRegularizationModule(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.params = []

    class Mode(enum.Enum):
        CLAMP = 0
        CYCLE = 1
        SCALE = 2

    class ParamToRegularize():
        def __init__(self, param, mode, limit_min, limit_max, init_min=None, init_max=None):
            if init_min is None:
                init_min = limit_min
            if init_max is None:
                init_max = limit_max
            self.param = param
            self.mode = mode
            self.limit_min = limit_min
            self.limit_max = limit_max
            self.init_min = init_min
            self.init_max = init_max

    def register_param_to_clamp(self, param, limit_min, limit_max, init_min=None, init_max=None):
        self.params.append(self.ParamToRegularize(param, self.Mode.CLAMP, limit_min, limit_max, init_min, init_max))

    def register_param_to_cycle(self, param, limit_min, limit_max, init_min=None, init_max=None):
        self.params.append(self.ParamToRegularize(param, self.Mode.CYCLE, limit_min, limit_max, init_min, init_max))

    def register_param_to_scale(self, param, limit_min, limit_max, init_min=None, init_max=None):
        self.params.append(self.ParamToRegularize(param, self.Mode.SCALE, limit_min, limit_max, init_min, init_max))

    def regularize_params(self):
        if hasattr(self, "pre_regularize"):
            self.pre_regularize()
        for item in self.params:
            if item.mode == self.Mode.CLAMP:
                item.param.data = torch.clamp(item.param.data, item.limit_min, item.limit_max)
            elif item.mode == self.Mode.CYCLE:
                item.param.data = item.param.data - item.limit_min
                item.param.data = item.param.data % (item.limit_max - item.limit_min)
                item.param.data = item.param.data + item.limit_min
            elif item.mode == self.Mode.SCALE:
                minimum = item.param.data.min()
                maximum = item.param.data.max()
                max_deviation = max(item.limit_min - minimum, maximum - item.limit_max)
                if max_deviation > 0.:
                    half_width = 0.5 * (item.limit_max - item.limit_min)
                    center = 0.5 * (item.limit_max + item.limit_min)
                    scale_factor = 1. / (max_deviation / half_width + 1)
                    item.param.data = scale_factor * (item.param.data - center) + center

    def intitialize_weights_uniform(self):
        for item in self.params:
            nn.init.uniform_(item.param, item.init_min, item.init_max)

    def intitialize_weights_zero(self):
        for item in self.params:
            nn.init.constant_(item.param, 0.)

    def intitialize_weights_identity(self):
        for item in self.params:
            nn.init.constant_(item.param, 1.)


class CopyModuleInterleave(nn.Module):
    def __init__(
        self,
        n_channels_in : int,
        n_channels_out : int,
    ) -> None:
        super().__init__()

        self.factor = n_channels_out // n_channels_in
        self.create_trackers(n_channels_in, n_channels_out)

    def create_trackers(self, n_channels_in, n_channels_out):
        self.tracker_out = tm.instance_tracker_module(label="Copy", draw_edges=True)
        out_channels_per_in_channel = n_channels_out // n_channels_in
        out_channel_inds = np.arange(n_channels_out)
        input_mapping = out_channel_inds // out_channels_per_in_channel
        self.tracker_out.register_data("input_mapping", input_mapping)

    def forward(self, x: Tensor) -> Tensor:
        out = x.repeat_interleave(self.factor, dim=1)
        _ = self.tracker_out(out)
        return out


class CopyModuleNonInterleave(nn.Module):
    def __init__(
        self,
        n_channels_in : int,
        n_channels_out : int,
    ) -> None:
        super().__init__()

        self.factor = n_channels_out // n_channels_in
        self.create_trackers(n_channels_in, n_channels_out)

    def create_trackers(self, n_channels_in, n_channels_out):
        self.tracker_out = tm.instance_tracker_module(label="Copy", draw_edges=True)
        out_channel_inds = np.arange(n_channels_out)
        input_mapping = out_channel_inds % n_channels_in
        self.tracker_out.register_data("input_mapping", input_mapping)

    def forward(self, x: Tensor) -> Tensor:
        dimensions = [1 for _ in range(x.ndim)]
        dimensions[1] = self.factor
        out =  x.repeat(*dimensions)
        _ = self.tracker_out(out)
        return out


class TrackedLeakyReLU(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.relu = nn.LeakyReLU(inplace=False)
        self.create_trackers()

    def create_trackers(self):
        self.tracker_out = tm.instance_tracker_module(label="Leaky ReLU")

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(x)
        _ = self.tracker_out(out)
        return out


class TrackedConv1x1(WeightRegularizationModule):
    def __init__(
        self, n_channels_in, n_channels_out, conv_groups,
    ) -> None:
        super().__init__()

        group_size = n_channels_in // conv_groups
        self.conv1x1 = nn.Conv2d(n_channels_in, n_channels_out, 1, 1, 0, groups=conv_groups, bias=False)
        self.register_param_to_scale(self.conv1x1.weight, -group_size, group_size, -1., 1.)
        self.create_trackers(n_channels_in, n_channels_out, conv_groups, group_size)

    def create_trackers(self, in_channels, out_channels, conv_groups, group_size):
        self.tracker_out = tm.instance_tracker_module(label="1x1 Conv", draw_edges=True, ignore_highlight=False)
        self.tracker_out.register_data("grouped_conv_weight", self.conv1x1.weight)
        self.tracker_out.register_data("grouped_conv_weight_limit", [-group_size, group_size])
        in_channels_per_group = in_channels // conv_groups
        out_channels_per_group = out_channels // conv_groups
        input_mapping = []
        for out_channel in range(out_channels):
            group = out_channel // out_channels_per_group
            input_mapping.append(list(range(group*in_channels_per_group, (group+1)*in_channels_per_group)))
        self.tracker_out.register_data("input_mapping", input_mapping)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1x1(x)
        _ = self.tracker_out(out)
        return out


class Conv1x1AndReLUModule(WeightRegularizationModule):
    def __init__(
        self,
        n_channels : int,
        conv_groups : int,
    ) -> None:
        super().__init__()

        self.conv1x1 = TrackedConv1x1(n_channels, n_channels, conv_groups)
        self.relu = TrackedLeakyReLU()

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.relu(x)
        return x


class BlendModule(WeightRegularizationModule):
    def __init__(
        self,
        n_channels : int,
        module_id_1 : int,
        module_id_2 : int,
        monitor_inputs : bool = False,
    ) -> None:
        super().__init__()
        
        w = torch.zeros(1, n_channels, 1, 1)
        self.weight = nn.Parameter(w, True)
        self.register_param_to_clamp(self.weight, 0., 1.)

        self.create_trackers(module_id_1, module_id_2, monitor_inputs)

    def create_trackers(self, module_id_1, module_id_2, monitor_inputs):
        if monitor_inputs:
            self.input_tracker_1 = tm.instance_tracker_module(label="Input A", precursors=[module_id_1])
            self.input_tracker_2 = tm.instance_tracker_module(label="Input B", precursors=[module_id_2])
            precursors = [self.input_tracker_1.module_id, self.input_tracker_2.module_id]
        else:
            self.input_tracker_1 = nn.Identity()
            self.input_tracker_2 = nn.Identity()
            precursors = [module_id_1, module_id_2]

        self.tracker_out = tm.instance_tracker_module(label="Blend", precursors=precursors, ignore_highlight=False)
        self.tracker_out.register_data("blend_weight", self.weight)
        self.tracker_out.register_data("blend_weight_limit", [0., 1.])

    def forward(self, x, y):
        _ = self.input_tracker_1(x)
        _ = self.input_tracker_2(y)
        out = self.weight * y + (1.0 - self.weight) * x
        _ = self.tracker_out(out)
        return out


class TrackedAvgPool2d(nn.Module):
    def __init__(
        self, kernel_size=2, stride=2, padding=0
    ) -> None:
        super().__init__()

        self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        self.create_trackers()

    def create_trackers(self):
        self.tracker_out = tm.instance_tracker_module(label="AvgPool", draw_edges=False, ignore_highlight=True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.pool(x)
        _ = self.tracker_out(out)
        return out


class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class NormalizationModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.create_trackers()

    def create_trackers(self):
        self.tracker_out = tm.instance_tracker_module(label="ScaleNorm")

    def forward(self, x: Tensor) -> Tensor:
        minimum = x.amin((1,2,3), keepdims=True).detach()
        maximum = x.amax((1,2,3), keepdims=True).detach()
        out = (x - minimum) / (maximum - minimum + 1e-6) # * 4. - 2. #I changed this for debug
        _ = self.tracker_out(out)
        return out


class LayerNorm(nn.Module):
    def __init__(self, _n_channels) -> None:
        super().__init__()

        self.create_trackers()

    def create_trackers(self):
        self.tracker_out = tm.instance_tracker_module(label="LayerNorm")

    def forward(self, x: Tensor) -> Tensor:
        dims = tuple(range(x.ndim))[1:]
        mean = x.mean(dim=dims, keepdims=True).detach()
        std = x.std(dim=dims, keepdims=True).detach()
        out = (x - mean) / (std + 1e-6)
        _ = self.tracker_out(out)
        return out
    

class PermutationModule(nn.Module):
    def __init__(self, indices):
        super().__init__()

        self.indices = nn.Parameter(indices, False)

        self.create_trackers()

    def create_trackers(self):
        self.tracker_out = tm.instance_tracker_module(label="Permutation", draw_edges=True)
        self.tracker_out.register_data("input_mapping", self.indices)
        self.tracker_out.register_data("indices", self.indices)

    def forward(self, x):
        x = x[:,self.indices]
        _ = self.tracker_out(x)
        return x


class ParameterizedFilterMode(enum.Enum):
   Even = 0
   Uneven = 1
   All = 2
   Random = 3
   Smooth = 4
   EvenPosOnly = 5
   UnevenPosOnly = 6
   TranslationSmooth = 7
   TranslationSharp4 = 8
   TranslationSharp8 = 8


class PredefinedConv(nn.Module):
    def __init__(self, n_channels_in: int, n_channels_out: int, stride: int = 1, padding : bool = True) -> None:
        super().__init__()

        self.padding = padding
        self.weight: nn.Parameter = None
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        assert self.n_channels_out >= self.n_channels_in
        self.stride = stride
        self.internal_weight = None # set by subclass

    # Called in init by subclass
    def create_trackers(self):
        self.tracker_out = tm.instance_tracker_module(label="PFModule")
        self.tracker_out.register_data("PFModule_kernels", self.weight)
    
    def forward(self, x: Tensor) -> Tensor:
        groups = self.n_channels_in
        if self.padding:
            x = F.pad(x, (1,1,1,1), "replicate")
        out = F.conv2d(x, self.internal_weight, None, self.stride, groups=groups)

        _ = self.tracker_out(out)
        #print(f"multadds {x.shape[2]*x.shape[3]*self.n_channels_out*self.weight.shape[1]*self.weight.shape[2]}")
        return out


def saddle(x, y, phi, sigma, uneven=True):
    a = np.arctan2(y, x)
    phi = np.deg2rad(phi)
    a = np.abs(phi - a)

    r = np.sqrt(x**2 + y**2)
    #b = np.sin(a) * r
    c = np.cos(a) * r

    if uneven:
        out = 1 - np.exp(-0.5*(c/sigma)**2)
        out[a>0.5*np.pi] = -out[a>0.5*np.pi]
    else:
        out = 2. * np.exp(-0.5*(c/sigma)**2)

    return out


def get_parameterized_filter(k: int=3, filter_mode: ParameterizedFilterMode=None, phi:float=0.):
    border = 0.5*(k-1.)
    x = np.linspace(-border, border, k)
    y = np.linspace(-border, border, k)
    xx, yy = np.meshgrid(x, y)
    if filter_mode==ParameterizedFilterMode.Even:
        data = saddle(xx, yy, phi, sigma=0.15*k, uneven=False)
        data = data - data.mean()
    elif filter_mode==ParameterizedFilterMode.Uneven:
        data = saddle(xx, yy, phi, sigma=0.3*k, uneven=True)
        data = data - data.mean()
    
    data = data / np.abs(data).sum()

    return data
    

class PredefinedConvnxn(PredefinedConv):
    def __init__(self, n_channels_in: int, n_channels_out: int, stride: int = 1, k: int = 3, filter_mode: ParameterizedFilterMode = ParameterizedFilterMode.All, n_angles: int = 4, requires_grad:bool=False, padding:bool=True) -> None:
        super().__init__(n_channels_in, n_channels_out, stride, padding)

        if requires_grad:
            raise NotImplementedError

        self.padding = k//2
        w = []
        if filter_mode == ParameterizedFilterMode.Uneven or filter_mode == ParameterizedFilterMode.All or filter_mode == ParameterizedFilterMode.UnevenPosOnly:
            w = w + [get_parameterized_filter(k, ParameterizedFilterMode.Uneven, phi) for phi in np.linspace(0, 180, n_angles, endpoint=False)]
        if filter_mode == ParameterizedFilterMode.Even or filter_mode == ParameterizedFilterMode.All or filter_mode == ParameterizedFilterMode.EvenPosOnly:
            w = w + [get_parameterized_filter(k, ParameterizedFilterMode.Even, phi) for phi in np.linspace(0, 180, n_angles, endpoint=False)]
        if filter_mode == ParameterizedFilterMode.TranslationSmooth:
            w = w + [get_parameterized_filter(k, ParameterizedFilterMode.Uneven, phi) for phi in np.linspace(0, 180, n_angles, endpoint=False)]
        if filter_mode == ParameterizedFilterMode.TranslationSharp4:
            w.append([
                [0.,0.,0.],
                [1.,0.,0.],
                [0.,0.,0.]]) # right
            w.append([
                [0.,1.,0.],
                [0.,0.,0.],
                [0.,0.,0.]]) # bottom
            w.append([
                [0.,0.,0.],
                [0.,0.,1.],
                [0.,0.,0.]]) # left
            w.append([
                [0.,0.,0.],
                [0.,0.,0.],
                [0.,1.,0.]]) # top
        elif filter_mode == ParameterizedFilterMode.TranslationSharp8:
            w.append([
                [0.,0.,0.],
                [1.,0.,0.],
                [0.,0.,0.]]) # right
            w.append([
                [0.,1.,0.],
                [0.,0.,0.],
                [0.,0.,0.]]) # bottom
            w.append([
                [0.,0.,0.],
                [0.,0.,1.],
                [0.,0.,0.]]) # left
            w.append([
                [0.,0.,0.],
                [0.,0.,0.],
                [0.,1.,0.]]) # top
            w.append([
                [1.,0.,0.],
                [0.,0.,0.],
                [0.,0.,0.]]) # bottom right
            w.append([
                [0.,0.,1.],
                [0.,0.,0.],
                [0.,0.,0.]]) # bottom left
            w.append([
                [0.,0.,0.],
                [0.,0.,0.],
                [0.,0.,1.]]) # top left
            w.append([
                [0.,0.,0.],
                [0.,0.,0.],
                [1.,0.,0.]]) # top right
        
        if filter_mode in [ParameterizedFilterMode.Even, ParameterizedFilterMode.Uneven, ParameterizedFilterMode.All, ParameterizedFilterMode.TranslationSmooth]:
            #w = [sign*item for item in w for sign in [-1, 1]]
            w.extend([-w_ for w_ in w])
        
        if filter_mode == ParameterizedFilterMode.Random:
            w = w + [np.random.rand(k, k) * 2. - 1. for _ in range(n_angles)]

        if filter_mode == ParameterizedFilterMode.TranslationSmooth:
            for w_ in w:
                w_[w_<0.] = 0.

        self.n_kernels = len(w)
        w = torch.FloatTensor(np.array(w))
        w = w.unsqueeze(1)
        self.weight = nn.Parameter(w, requires_grad)
        n_channels_per_kernel = self.n_channels_out // self.n_kernels
        internal_weight = self.weight.data.repeat((n_channels_per_kernel, 1, 1, 1)) # This involves copying. Here we compute internal weights only once. This means that the filters are not trainable. For trainable version, the internal weights have to be recomputed in the forward function.
        self.internal_weight = nn.Parameter(internal_weight, False)

        self.create_trackers()
    

class PredefinedFilterModule3x3Part(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        filter_mode: ParameterizedFilterMode,
        n_angles: int,
        handcrafted_filters_require_grad: bool = False,
        f: int = 1,
        k: int = 3,
        stride: int = 1,
        padding: bool = True,
    ) -> None:
        super().__init__()

        # Copy. This copy module is only decoration. The consecutive module will not be affected by copy.
        self.copymodule = CopyModuleInterleave(n_channels_in, n_channels_in * f) if f>1 else nn.Identity()
        self.predev_conv = PredefinedConvnxn(n_channels_in, n_channels_in * f, stride=stride, k=k, filter_mode=filter_mode, n_angles=n_angles, requires_grad=handcrafted_filters_require_grad, padding=padding)
        self.activation_layer = TrackedLeakyReLU()


    def forward(self, x: Tensor) -> Tensor:
        _ = self.copymodule(x)
        x = self.predev_conv(x)
        x = self.activation_layer(x)
        return x


def zip_tensors(tensors):
    shape = tensors[0].shape
    tensors = [x.unsqueeze(2) for x in tensors]
    out = torch.cat(tensors, 2)
    out = out.reshape((shape[0], -1, shape[2], shape[3]))
    return out


class RandomRoll(nn.Module):
    def __init__(
        self,
        shift : int = 1,
    ) -> None:
        super().__init__()

        self.shift = shift

        self.create_trackers()

    def create_trackers(self):
        tm.instance_tracker_module_group(label="RandomRoll")
        self.tracker_in = tm.instance_tracker_module(label="Input")
        self.tracker_out = tm.instance_tracker_module(label="RandomRoll")

    def forward(self, x):
        _ = self.tracker_in(x)
        # has to match the filter combinations in 3x3 part
        random_roll_array = [self.shift, self.shift, -self.shift, -self.shift]
        random.shuffle(random_roll_array)
        direction_array = [2,2,3,3]
        random.shuffle(direction_array)

        x1 = torch.roll(x[:,0::4], random_roll_array[0], direction_array[0]) # right
        x2 = torch.roll(x[:,1::4], random_roll_array[1], direction_array[1]) # bottom
        x3 = torch.roll(x[:,2::4], random_roll_array[2], direction_array[2]) # left
        x4 = torch.roll(x[:,3::4], random_roll_array[3], direction_array[3]) # top
        x_stacked = zip_tensors([x1, x2, x3, x4])
        _ = self.tracker_out(x_stacked)
        return x_stacked


class ParamTranslationModule(WeightRegularizationModule):
    def __init__(
        self,
        n_channels : int = 1,
        k : int = 3,
        dim : int = 2,
        spatial_requires_grad : bool = True,

    ) -> None:
        super().__init__()

        # Initialize trainable internal weights
        weight = torch.ones(n_channels, 1, 1, 1) * 0.49 # must not be 0.5 because the torch.abs(x) function in forward has no gradient for x=0
        self.weight = nn.Parameter(weight, spatial_requires_grad)
        self.register_param_to_clamp(self.weight, 0., 1.)

        # Initialize helpers
        kernel_positions = torch.zeros(1,1,k)
        for i in range(k):
                kernel_positions[:,:,i] = i/(k-1.)
        if dim==2:
            kernel_positions = kernel_positions.unsqueeze(3)
        else:
            kernel_positions = kernel_positions.unsqueeze(2)
        self.kernel_positions = nn.Parameter(kernel_positions, False)
        self.radius = 1./(k-1.)

        self.create_trackers(dim)

    def create_trackers(self, dim):
        mode = "H" if dim==3 else "V"
        self.tracker_out = tm.instance_tracker_module(label=f"Transl{mode}", ignore_highlight=True)
        self.tracker_out.register_data("weight_per_channel", self.weight)
        self.tracker_out.register_data("weight_per_channel_limit", [0., 1.])

    def forward(self, x):
        w = - torch.abs(self.weight - self.kernel_positions) / self.radius + 1.
        w = DifferentiableClamp.apply(w, 0., 1.)
        out = F.conv2d(x, w, groups=x.shape[1])
        _ = self.tracker_out(out)
        return out


    def intitialize_weights_alternating(self):
        for item in self.params_to_clamp:
            data = item['param'].data
            shape = data.shape
            data = data.flatten()
            data [0::4] = item['init_max']
            data [1::4] = 0.49
            data [2::4] = item['init_min']
            data [3::2] = 0.49
            data = data.reshape(shape)
            item['param'].data = data


    def intitialize_weights_antialternating(self):
        for item in self.params_to_clamp:
            data = item['param'].data
            shape = data.shape
            data = data.flatten()
            data [0::4] = 0.49
            data [1::4] = item['init_max']
            data [2::4] = 0.49
            data [3::2] = item['init_min']
            data = data.reshape(shape)
            item['param'].data = data


class ParamTranslationGroup(nn.Module):
    def __init__(
        self,
        n_channels : int = 1,
        k : int = 3,
        spatial_requires_grad : bool = True,

    ) -> None:
        super().__init__()
        
        self.padding = k//2
        self.roll_v = ParamTranslationModule(n_channels, k, 2, spatial_requires_grad)
        self.roll_h = ParamTranslationModule(n_channels, k, 3, spatial_requires_grad)

    def forward(self, x):
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), "replicate")
        x = self.roll_v(x)
        x = self.roll_h(x)
        return x
        

class PreprocessingModule(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        conv_groups: int = 1,
        avgpool: bool = True,
        norm_module : nn.Module = LayerNorm,
        permutation : str = "shifted" # one of shifted, identity, disabled
    ) -> None:
        super().__init__()

        tm.instance_tracker_module_group(label="Preprocessing")

        # Input tracker
        self.tracker_in = tm.instance_tracker_module(label="Input")

        # Pooling
        self.avgpool = TrackedAvgPool2d(kernel_size=2, stride=2, padding=0) if avgpool else nn.Identity()

        # Permutation
        group_size = n_channels_in // conv_groups
        if permutation == "shifted":
            self.permutation_module = PermutationModule(torch.arange(n_channels_in).roll(group_size // 2))
        elif permutation == "identity":
            self.permutation_module = PermutationModule(torch.arange(n_channels_in))
        elif permutation == "disabled":
            self.permutation_module = nn.Identity()
        else:
            raise ValueError

        # Copy channels.
        if n_channels_in != n_channels_out:
            self.copymodule = CopyModuleNonInterleave(n_channels_in, n_channels_out)
        else:
            self.copymodule = nn.Identity()
            
        # Norm
        self.norm_module = norm_module(n_channels_out)

    def forward(self, x: Tensor) -> Tensor:
        _ = self.tracker_in(x)
        x = self.avgpool(x)
        x = self.permutation_module(x)
        x = self.copymodule(x)
        x = self.norm_module(x)
        return x


class TranslationBlock(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int, # n_channels_out % shuffle_conv_groups == 0
        conv_groups: int = 1,
        avgpool: bool = True,
        spatial_mode : str = "predefined_filters", # one of predefined_filters and parameterized_translation
        spatial_requires_grad : bool = True,
        filter_mode: str = "Uneven",
        n_angles : int = 2,
        translation_k : int = 3,
        randomroll: int = -1,
        normalization_mode : str = "layernorm", # one of batchnorm, layernorm
        permutation : str = "shifted", # one of shifted, identity, disabled
    ) -> None:
        super().__init__()

        if normalization_mode == "layernorm":
            norm_module = LayerNorm
        elif normalization_mode == "batchnorm":
            norm_module = nn.BatchNorm2d
        else:
            raise ValueError

        self.preprocessing = PreprocessingModule(n_channels_in, n_channels_out, conv_groups, avgpool, norm_module, permutation)

        # 1x1 Conv
        tm.instance_tracker_module_group(label="1x1 Conv")
        self.tracker_input_conv_1 = tm.instance_tracker_module(label="Input")
        self.conv1x1_1 = Conv1x1AndReLUModule(n_channels_out, conv_groups)

        # Random roll (attack)
        self.randomroll = RandomRoll(randomroll) if randomroll>0 else nn.Identity()
        
        # Spatial operation
        if spatial_mode == "predefined_filters":
            tm.instance_tracker_module_group(label="3x3 Conv")
            self.tracker_input_spatial = tm.instance_tracker_module(label="Input")
            self.spatial = PredefinedFilterModule3x3Part(
                n_channels_in=n_channels_out,
                filter_mode=ParameterizedFilterMode[filter_mode],
                n_angles=n_angles,
                handcrafted_filters_require_grad=spatial_requires_grad,
                f=1,
                k=3,
                stride=1,
            )
        elif spatial_mode == "parameterized_translation":
            tm.instance_tracker_module_group(label="Translation")
            self.tracker_input_spatial = tm.instance_tracker_module(label="Input")
            self.spatial = ParamTranslationGroup(n_channels_out, translation_k, spatial_requires_grad)
        # Spatial blending (skip)
        self.blend = BlendModule(n_channels_out, self.tracker_input_spatial.module_id, tm.module_id, monitor_inputs=False)

        # 1x1 Conv
        tm.instance_tracker_module_group(label="1x1 Conv")
        self.tracker_input_conv_2 = tm.instance_tracker_module(label="Input")
        self.conv1x1_2 = Conv1x1AndReLUModule(n_channels_out, conv_groups)


    def forward(self, x: Tensor) -> Tensor:
        x = self.preprocessing(x)
        _ = self.tracker_input_conv_1(x)
        x = self.conv1x1_1(x)
        x = self.randomroll(x)
        x_skip = x
        _ = self.tracker_input_spatial(x)
        x = self.spatial(x)
        x = self.blend(x_skip, x)
        _ = self.tracker_input_conv_2(x)
        x = self.conv1x1_2(x)
        return x


def initialize_weights(modules, init_mode):
    for m in modules:
        if isinstance(m, WeightRegularizationModule):
            if init_mode in ['uniform', 'uniform_translation_as_pfm'] :
                m.intitialize_weights_uniform()
            elif init_mode == 'zero':
                m.intitialize_weights_zero()
            elif init_mode == 'identity':
                m.intitialize_weights_identity()
        if isinstance(m, ParamTranslationModule):
            if init_mode == 'uniform_translation_as_pfm':
                if m.kernel_positions.shape[3] >> 1:
                    m.intitialize_weights_alternating()
                else:
                    m.intitialize_weights_antialternating()