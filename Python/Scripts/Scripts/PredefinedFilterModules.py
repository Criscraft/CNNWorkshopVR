import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import enum

from TrackingModules import TrackerModuleProvider

"""
PredefinedConv padding is zero padding instead of replicate. This seems to improve test accuracy by about 1%.
"""

tm = TrackerModuleProvider()

class ChannelPadding(nn.Module):
    def __init__(
        self,
        n_channels_in : int,
        n_channels_out : int,
    ) -> None:
        super().__init__()

        self.padding_size = n_channels_out - n_channels_in
        assert(self.padding_size > 0)

    def forward(self, x):
        shape = list(x.shape)
        shape[1] = self.padding_size
        padding = torch.zeros(shape, device=x.device)
        return torch.cat((x, padding), 1)
    

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

        #self.relu = nn.ReLU(inplace=False)
        self.relu = nn.LeakyReLU(inplace=False)
        self.create_trackers()

    def create_trackers(self):
        self.tracker_out = tm.instance_tracker_module(label="Leaky ReLU")

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(x)
        _ = self.tracker_out(out)
        return out


class TrackedConv1x1Regularized(WeightRegularizationModule):
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
        n_channels_in : int,
        n_channels_out : int,
        conv_groups : int,
    ) -> None:
        super().__init__()

        self.conv1x1 = TrackedConv1x1Regularized(n_channels_in, n_channels_out, conv_groups)
        self.relu = TrackedLeakyReLU()

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.relu(x)
        return x
    

class TrackedConv(nn.Module):
    def __init__(
        self,
        n_channels_in : int,
        n_channels_out : int,
        conv_groups : int = 1,
        k : int = 3,
        stride : int = 1,
        padding=0,
        bias : bool = True
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(n_channels_in, n_channels_out, k, padding=padding, padding_mode='replicate', stride=stride, groups=conv_groups, bias=bias)
        self.create_trackers(n_channels_in, n_channels_out, k)

    
    def create_trackers(self, n_channels_in, n_channels_out, k):
        self.tracker_out = tm.instance_tracker_module(label="Conv")
        self.tracker_out.register_data("display", f"weight: {n_channels_out}x{n_channels_in}x{k}x{k}")
                

    def forward(self, x):
        x = self.conv(x)
        _ = self.tracker_out(x)
        return x
    

class TrackedLinear(nn.Module):
    def __init__(
        self,
        n_channels_in : int,
        n_channels_out : int,
        k : int = 3,
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(n_channels_in, n_channels_out)
        self.create_trackers(n_channels_in, n_channels_out, k)

    
    def create_trackers(self, n_channels_in, n_channels_out, k):
        self.tracker_out = tm.instance_tracker_module(label="Linear")
        self.tracker_out.register_data("display", f"weight: {n_channels_out}x{n_channels_in}")
                

    def forward(self, x):
        x = self.linear(x)
        _ = self.tracker_out(x.unsqueeze(2).unsqueeze(3))
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


class Interpolate(nn.Module):
    def __init__(self, antialias=False):
        super().__init__()
        self.antialias = antialias

    def forward(self, x):
        return F.interpolate(x, scale_factor=0.5, mode='bilinear', antialias=self.antialias)
    

class Subsample(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:,:,::2,::2]


class HardSmoothConv(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.padding = 1

        w = [[[[1.,1.],
            [1.,1.]]]]
        w = torch.FloatTensor(w)
        #w = w.unsqueeze(1)
        self.w = nn.Parameter(w, False)


    def forward(self, x: Tensor) -> Tensor:
        self.to(x.device)
        n_channels_in = x.shape[1]
        w_tmp = self.w.repeat((n_channels_in, 1, 1, 1))
        out = F.conv2d(x, w_tmp, padding=self.padding, groups=n_channels_in)
        return out
    

class GlobalLPPool(nn.Module):
    def __init__(self, p : int = 2):
        super().__init__()
        self.p = p

    def forward(self, x):
        out = x.pow(self.p).sum((2,3), keepdims=True)
        out = out.sign() * (F.relu(out.abs()) * x.shape[2] * x.shape[3]).pow(1.0 / self.p)
        return out
    

class TrackedPool(nn.Module):
    def __init__(self, 
        pool_mode : str = "avgpool",
        k : int = 2,
    ):
        super().__init__()
        self.k = k
        self.pool = None
        self.set_tracked_pool_mode(pool_mode)
        self.create_trackers()

    def set_tracked_pool_mode(self, pool_mode : str):
        if pool_mode == "avgpool":
            self.pool = nn.AvgPool2d(kernel_size=self.k, stride=2, padding=self.k//2)
            print("set to avgpool")
        elif pool_mode == "maxpool":
            self.pool = nn.MaxPool2d(kernel_size=self.k, stride=2, padding=self.k//2)
            print("set to maxpool")
        elif pool_mode == "interpolate_antialias":
            self.pool = Interpolate(True)
            print("set to interpolate_antialias")
        elif pool_mode == "interpolate":
            self.pool = Interpolate(False)
            print("set to interpolate")
        elif pool_mode == "subsample":
            self.pool = Subsample()
            print("set to subsample")
        elif pool_mode == "identity":
            self.pool = nn.Identity()
            print("set to identity")
        elif pool_mode == "identity_smooth":
            self.pool = HardSmoothConv()
            print("set to identity_smooth")
        elif pool_mode == "lppool":
            self.pool = nn.LPPool2d(kernel_size=2, stride=2, norm_type=2)
            print("set to lppool")

    def create_trackers(self):
        self.tracker_out = tm.instance_tracker_module(label="Pooling", draw_edges=False, ignore_highlight=True)

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


class TrackedLayerNorm(nn.Module):
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
    

class TrackedBatchNorm(nn.Module):
    def __init__(self, _n_channels) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(_n_channels)
        self.create_trackers()

    def create_trackers(self):
        self.tracker_out = tm.instance_tracker_module(label="BatchNorm")

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        _ = self.tracker_out(x)
        return x
    

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
   EvenAndUneven = 2
   Random = 3
   Smooth = 4
   EvenPosOnly = 5
   UnevenPosOnly = 6
   TranslationSmooth = 7
   TranslationSharp4 = 8
   TranslationSharp8 = 9
   HardSmooth2x2 = 10


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


def smooth(x, y, sigma):
    r = np.sqrt(x**2 + y**2)
    out =  np.exp(-0.5*(r/sigma)**2)
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
    elif filter_mode==ParameterizedFilterMode.Smooth:
        data = smooth(xx, yy, sigma=0.25*k)
    
    data = data / np.abs(data).sum()

    return data
    

class PredefinedConv(nn.Module):
    def __init__(self, n_channels_in: int, n_channels_out: int, stride: int = 1, k: int = 3, filter_mode: ParameterizedFilterMode = ParameterizedFilterMode.EvenAndUneven, n_angles: int = 4, filters_require_grad:bool=False, padding:bool=True) -> None:
        super().__init__()

        assert n_channels_out >= n_channels_in
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.stride = stride
        self.padding = k // 2 if padding else 0
        self.dilation = 1

        self.filters_require_grad = filters_require_grad
        
        w = []
        if filter_mode == ParameterizedFilterMode.Uneven or filter_mode == ParameterizedFilterMode.EvenAndUneven or filter_mode == ParameterizedFilterMode.UnevenPosOnly:
            w = w + [get_parameterized_filter(k, ParameterizedFilterMode.Uneven, phi) for phi in np.linspace(0, 180, n_angles, endpoint=False)]
        if filter_mode == ParameterizedFilterMode.Even or filter_mode == ParameterizedFilterMode.EvenAndUneven or filter_mode == ParameterizedFilterMode.EvenPosOnly:
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
        elif filter_mode==ParameterizedFilterMode.HardSmooth2x2:
            w.append([
                [1.,1.],
                [1.,1.]])
        
        if filter_mode in [ParameterizedFilterMode.Even, ParameterizedFilterMode.Uneven, ParameterizedFilterMode.EvenAndUneven, ParameterizedFilterMode.TranslationSmooth]:
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
        self.weight = nn.Parameter(w, self.filters_require_grad)
        n_channels_per_kernel = self.n_channels_out // self.n_kernels
        internal_weight = self.weight.data.repeat((n_channels_per_kernel, 1, 1, 1)) # This involves copying. Here we compute internal weights only once. This means that the filters are not trainable. For trainable version, the internal weights have to be recomputed in the forward function.
        self.internal_weight = nn.Parameter(internal_weight, False)

        self.create_trackers()


    def resize_filter_to_mimic_poolstage(self, pool_stage):
        # n_channels_per_kernel = self.n_channels_out // self.n_kernels
        # if pool_stage == 0:
        #     weight = self.weight.data.repeat((n_channels_per_kernel, 1, 1, 1))
        # else:
        #     expand_factor = 2**pool_stage
        #     weight = self.weight.data.clone().detach()
        #     weight = weight.repeat_interleave(expand_factor,1).repeat_interleave(expand_factor,2)
        # self.internal_weight.data = weight.data.repeat((n_channels_per_kernel, 1, 1, 1))
        expand_factor = 2**pool_stage
        self.dilation = expand_factor
        effective_kernel_size = 1+2*expand_factor
        self.padding = effective_kernel_size//2


    def create_trackers(self):
        self.tracker_out = tm.instance_tracker_module(label="PFModule")
        self.tracker_out.register_data("PFModule_kernels", self.weight)


    def update_internal_weights(self):
        # The internal weights have to be recomputed when the weights have been changed.
        n_channels_per_kernel = self.n_channels_out // self.n_kernels
        self.internal_weight.data = self.weight.data.repeat((n_channels_per_kernel, 1, 1, 1))
    

    def forward(self, x: Tensor) -> Tensor:
        groups = self.n_channels_in
        # if self.padding > 0:
        #     x = F.pad(x, (self.padding,self.padding,self.padding,self.padding), "replicate")
        out = F.conv2d(x, self.internal_weight, None, self.stride, groups=groups, dilation=self.dilation, padding=self.padding)

        _ = self.tracker_out(out)
        #print(f"multadds {x.shape[2]*x.shape[3]*self.n_channels_out*self.weight.shape[1]*self.weight.shape[2]}")
        return out


class TrackedSmoothConv(nn.Module):
    def __init__(
        self,
        n_channels : int,
        k : int = 3,
    ) -> None:
        super().__init__()

        self.padding = k//2
        self.n_channels = n_channels

        w = [get_parameterized_filter(k, ParameterizedFilterMode.Smooth)]
        w = torch.FloatTensor(w)
        w = w.unsqueeze(1)
        
        internal_weight = w.repeat((n_channels, 1, 1, 1))
        self.internal_weight = nn.Parameter(internal_weight, False)

        self.create_trackers()

    
    def create_trackers(self):
        self.tracker_out = tm.instance_tracker_module(label="Blurring")


    def forward(self, x: Tensor) -> Tensor:
        out = F.conv2d(x, self.internal_weight, None, 1, self.padding, groups=self.n_channels)
        _ = self.tracker_out(out)
        return out
        

class PredefinedConvWithDecorativeCopy(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        filter_mode: ParameterizedFilterMode,
        n_angles: int,
        filters_require_grad: bool = False,
        f: int = 1,
        k: int = 3,
        stride: int = 1,
        padding: bool = True,
    ) -> None:
        super().__init__()

        # Copy. This copy module is only decoration. The consecutive module will not be affected by copy.
        self.copymodule = CopyModuleInterleave(n_channels_in, n_channels_in * f) if f>1 else nn.Identity()
        self.predev_conv = PredefinedConv(n_channels_in, n_channels_in * f, stride=stride, k=k, filter_mode=filter_mode, n_angles=n_angles, filters_require_grad=filters_require_grad, padding=padding)


    def forward(self, x: Tensor) -> Tensor:
        # This copy module is only decoration. The consecutive module will not be affected by copy.
        _ = self.copymodule(x)
        x = self.predev_conv(x)
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
        #tm.instance_tracker_module_group(label="RandomRoll")
        #self.tracker_in = tm.instance_tracker_module(label="Input")
        self.tracker_out = tm.instance_tracker_module(label="RandomRoll")

    def forward(self, x):
        n = x.shape[1]
        indices = torch.randperm(n, device=x.device)
        indices_rev = indices.argsort()

        x1 = torch.roll(x[:, indices[0*n//4 : 1*n//4]], self.shift, 2) # bottom
        x2 = torch.roll(x[:, indices[1*n//4 : 2*n//4]], -self.shift, 2) # top
        x3 = torch.roll(x[:, indices[2*n//4 : 3*n//4]], self.shift, 3) # right
        x4 = torch.roll(x[:, indices[3*n//4 : 4*n//4]], -self.shift, 3) # left
        x_stacked = torch.cat([x1, x2, x3, x4], 1)
        x_stacked = x_stacked[:, indices_rev]
        _ = self.tracker_out(x_stacked)
        return x_stacked


class ParamTranslationModule(WeightRegularizationModule):
    def __init__(
        self,
        n_channels : int = 1,
        k : int = 3, # needs to be odd
        dim : int = 2,
        gradient_radius : float = 0.,
        filters_require_grad : bool = True,

    ) -> None:
        super().__init__()
        assert k % 2 == 1
        self.k = k
        self.filters_require_grad = filters_require_grad
        self.gradient_radius = gradient_radius

        # Initialize trainable internal weights
        weight = (torch.rand(n_channels, 1, 1, 1) - 0.5) * 2. # must not be 0.0 because the torch.abs(x) function in forward has no gradient for x=0
        self.weight = nn.Parameter(weight, filters_require_grad)
        self.register_param_to_clamp(self.weight, -1., 1.)

        # Initialize helpers
        self.step_size = 2./(k-1.)
        kernel_positions = torch.zeros(1,1,k)
        for i in range(k):
                kernel_positions[:,:,i] = - 1. + i * self.step_size
        if dim==2:
            kernel_positions = kernel_positions.unsqueeze(3)
        else:
            kernel_positions = kernel_positions.unsqueeze(2)
        self.kernel_positions = nn.Parameter(kernel_positions, False)

        internal_weight = self.get_internal_weights()
        self.internal_weight = nn.Parameter(internal_weight, False)
        
        self.create_trackers(dim)

    def create_trackers(self, dim):
        mode = "H" if dim==3 else "V"
        self.tracker_out = tm.instance_tracker_module(label=f"Transl{mode}", ignore_highlight=True)
        self.tracker_out.register_data("weight_per_channel", self.weight)
        self.tracker_out.register_data("weight_per_channel_limit", [-1., 1.])

    def get_internal_weights(self):
        # The internal weights have to be recomputed after any change of the weights
        internal_weight = - torch.abs(self.weight - self.kernel_positions) / self.step_size + 1.
        internal_weight[internal_weight < -self.gradient_radius] = 0.
        internal_weight = DifferentiableClamp.apply(internal_weight, 0., 1.)
        return internal_weight
    
    def update_internal_weights(self):
       self.internal_weight.data = self.get_internal_weights()

    def forward(self, x):
        if self.training:
            internal_weight = self.get_internal_weights()
        else:
            internal_weight = self.internal_weight
        out = F.conv2d(x, internal_weight, groups=x.shape[1])
        _ = self.tracker_out(out)
        return out


    def intitialize_weights_alternating(self):
        for item in self.params_to_clamp:
            data = item['param'].data
            shape = data.shape
            data = data.flatten()
            data [0::4] = item['init_max']
            data [1::4] = 0.01
            data [2::4] = item['init_min']
            data [3::2] = 0.01
            data = data.reshape(shape)
            item['param'].data = data


    def intitialize_weights_antialternating(self):
        for item in self.params_to_clamp:
            data = item['param'].data
            shape = data.shape
            data = data.flatten()
            data [0::4] = 0.01
            data [1::4] = item['init_max']
            data [2::4] = 0.01
            data [3::2] = item['init_min']
            data = data.reshape(shape)
            item['param'].data = data


class ParamTranslationGroup(nn.Module):
    def __init__(
        self,
        n_channels : int = 1,
        k : int = 3,
        gradient_radius : float = 0.,
        filters_require_grad : bool = True,

    ) -> None:
        super().__init__()
        
        self.padding = k//2
        self.roll_v = ParamTranslationModule(n_channels, k, 2, gradient_radius, filters_require_grad)
        self.roll_h = ParamTranslationModule(n_channels, k, 3, gradient_radius, filters_require_grad)

    def forward(self, x):
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding))
        x = self.roll_v(x)
        x = self.roll_h(x)
        return x


def initialize_weights(modules, init_mode):
    for m in modules:
        if isinstance(m, WeightRegularizationModule):
            if init_mode in ['uniform', 'uniform_translation_as_pfm']:
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
        # kaiming initialization resulted in about 1% less accuracy compared to the default init.
        # if isinstance(m, nn.Conv2d):
        #     if init_mode in ['kaiming']:
        #         nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
