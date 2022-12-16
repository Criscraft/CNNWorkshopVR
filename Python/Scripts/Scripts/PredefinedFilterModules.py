import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import enum
import random

from TrackingModules import TrackerModuleProvider

tm = TrackerModuleProvider()

class CopyModule(nn.Module):
    def __init__(
        self,
        n_channels_in : int,
        n_channels_out : int,
        interleave : bool = True,
    ) -> None:
        super().__init__()

        self.factor = n_channels_out // n_channels_in
        self.interleave = interleave
        info_code = "interleave" if self.interleave else ""
        self.tracker_copymodule = tm.instance_tracker_module(label="Copy", info_code=info_code)

    def forward(self, x: Tensor) -> Tensor:
        if self.interleave:
            out = x.repeat_interleave(self.factor, dim=1)
        else:
            dimensions = [1 for _ in range(x.ndim)]
            dimensions[1] = self.factor
            out =  x.repeat(*dimensions)
        
        _ = self.tracker_copymodule(out)
        
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

        self.tracker_norm = tm.instance_tracker_module(label="ScaleNorm")

    def forward(self, x: Tensor) -> Tensor:
        minimum = x.amin((1,2,3), keepdims=True).detach()
        maximum = x.amax((1,2,3), keepdims=True).detach()
        out = (x - minimum) / (maximum - minimum + 1e-6) # * 4. - 2. #I changed this for debug
        _ = self.tracker_norm(out)
        return out


class LayerNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.tracker_norm = tm.instance_tracker_module(label="LayerNorm")

    def forward(self, x: Tensor) -> Tensor:
        dims = tuple(range(x.ndim))[1:]
        mean = x.mean(dim=dims, keepdims=True).detach()
        std = x.std(dim=dims, keepdims=True).detach()
        out = (x - mean) / (std + 1e-6)
        _ = self.tracker_norm(out)
        return out
    

class PermutationModule(nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.indices = nn.Parameter(indices, False)
        self.tracker_permutation = tm.instance_tracker_module(label="Permutation", tracked_module=self)

    def forward(self, x):
        x = x[:,self.indices]
        _ = self.tracker_permutation(x)
        return x
    

class ParameterizedFilterMode(enum.Enum):
   Even = 0
   Uneven = 1
   All = 2
   Random = 3
   Smooth = 4
   EvenPosOnly = 5
   UnevenPosOnly = 6
   Translation = 7


class PredefinedConv(nn.Module):
    def __init__(self, n_channels_in: int, n_channels_out: int, stride: int = 1, padding : bool = True) -> None:
        super().__init__()

        self.padding = padding
        self.weight: nn.Parameter = None
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.stride = stride
        assert self.n_channels_out >= self.n_channels_in

    
    def forward(self, x: Tensor) -> Tensor:
        n_channels_per_kernel = self.n_channels_out // self.n_kernels

        w_tmp = self.weight.repeat((n_channels_per_kernel, 1, 1, 1)) # this involves copying
        groups = self.n_channels_in

        if self.padding:
            x = F.pad(x, (1,1,1,1), "replicate")
        out = F.conv2d(x, w_tmp, None, self.stride, groups=groups)
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

        self.padding = k//2
        w = []
        if filter_mode == ParameterizedFilterMode.Uneven or filter_mode == ParameterizedFilterMode.All or filter_mode == ParameterizedFilterMode.UnevenPosOnly or filter_mode == ParameterizedFilterMode.Translation:
            w = w + [get_parameterized_filter(k, ParameterizedFilterMode.Uneven, phi) for phi in np.linspace(0, 180, n_angles, endpoint=False)]
        if filter_mode == ParameterizedFilterMode.Even or filter_mode == ParameterizedFilterMode.All or filter_mode == ParameterizedFilterMode.EvenPosOnly:
            w = w + [get_parameterized_filter(k, ParameterizedFilterMode.Even, phi) for phi in np.linspace(0, 180, n_angles, endpoint=False)]
        
        if not filter_mode == ParameterizedFilterMode.UnevenPosOnly and not filter_mode == ParameterizedFilterMode.EvenPosOnly:
            #w = [sign*item for item in w for sign in [-1, 1]]
            w.extend([-w_ for w_ in w])
        
        if filter_mode == ParameterizedFilterMode.Random:
            w = w + [np.random.rand(k, k) * 2. - 1. for _ in range(n_angles)]

        if filter_mode == ParameterizedFilterMode.Translation:
            for w_ in w:
                w_[w_<0.] = 0.

        self.n_kernels = len(w)
        w = torch.FloatTensor(np.array(w))
        w = w.unsqueeze(1)
        self.weight = nn.Parameter(w, requires_grad)
    

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
        activation_layer : nn.Module = nn.ReLU,
        padding: bool = True,
    ) -> None:
        super().__init__()

        # Copy. This copy module is only decoration. The consecutive module will not be affected by copy.
        self.copymodule = CopyModule(n_channels_in, n_channels_in * f) if f>1 else nn.Identity()
        
        self.predev_conv = PredefinedConvnxn(n_channels_in, n_channels_in * f, stride=stride, k=k, filter_mode=filter_mode, n_angles=n_angles, requires_grad=handcrafted_filters_require_grad, padding=padding)
        self.tracker_predev_conv = tm.instance_tracker_module(label="3x3 Conv", tracked_module=self.predev_conv)
        
        self.activation_layer = activation_layer(inplace=False)
        if not isinstance(self.activation_layer, nn.Identity):
            self.activation_layer_tracker = tm.instance_tracker_module(label="ReLU")
        else:
            self.activation_layer_tracker = nn.Identity()


    def forward(self, x: Tensor) -> Tensor:
        _ = self.copymodule(x)
        x = self.predev_conv(x)
        _ = self.tracker_predev_conv(x)
        x = self.activation_layer(x)
        _ = self.activation_layer_tracker(x)
        return x


class MixGroup(nn.Module):
    def __init__(
        self,
        n_channels : int,
        conv_groups : int,
    ) -> None:
        super().__init__()

        self.tracker_input = tm.instance_tracker_module(label="Input")
        self.conv1x1 = nn.Conv2d(n_channels, n_channels, 1, 1, 0, groups=conv_groups, bias=False)
        #group_size = n_channels // conv_groups
        self.conv1x1.weight_limit_min = -1.
        self.conv1x1.weight_limit_max = 1.
        self.conv1x1_tracker = tm.instance_tracker_module(label="1x1 Conv", tracked_module=self.conv1x1)
        self.relu = nn.LeakyReLU(inplace=False)
        self.relu_tracker = tm.instance_tracker_module(label="ReLU")

    def forward(self, x):
        _ = self.tracker_input(x)
        x = self.conv1x1(x)
        _ = self.conv1x1_tracker(x)
        x = self.relu(x)
        _ = self.relu_tracker(x)
        return x


class MergeModule(nn.Module):
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
        self.weight_limit_min = 0.
        self.weight_limit_max = 1.
        

        if monitor_inputs:
            self.input_tracker_1 = tm.instance_tracker_module(label="Input A", precursors=[module_id_1])
            self.input_tracker_2 = tm.instance_tracker_module(label="Input B", precursors=[module_id_2])
            self.tracker_merge = tm.instance_tracker_module(label="Merge", tracked_module=self, precursors=[self.input_tracker_1.module_id, self.input_tracker_2.module_id])
        else:
            self.input_tracker_1 = nn.Identity()
            self.input_tracker_2 = nn.Identity()
            self.tracker_merge = tm.instance_tracker_module(label="Merge", tracked_module=self, precursors=[module_id_1, module_id_2])

    def forward(self, x, y):
        _ = self.input_tracker_1(x)
        _ = self.input_tracker_2(y)
        out = self.weight * y + (1.0 - self.weight) * x
        _ = self.tracker_merge(out)
        return out


class MergeModuleHiddenSymmetric(nn.Module):
    def __init__(
        self,
        weight_pointer,
    ) -> None:
        super().__init__()

        self.weight = weight_pointer

    def forward(self, x, y):
        w = torch.abs(self.weight - 0.5) * 2.0
        out = w * y + (1.0 - w) * x
        return out


class RollModuleHParam(nn.Module):
    def __init__(
        self,
        n_channels : int,
        roll_range : int = 1,
    ) -> None:
        super().__init__()
        
        w = torch.ones(1, n_channels, 1, 1) * 0.5
        self.weight = nn.Parameter(w, True)
        self.weight_limit_min = 0.
        self.weight_limit_max = 1.
        self.tracker_roll = tm.instance_tracker_module(label="Roll H", tracked_module=self, info_code="pass_highlight")
        self.merge = MergeModuleHiddenSymmetric(self.weight)
        #self.merge = MergeModule(n_channels, self.tracker_roll.module_id, self.tracker_roll.module_id) # this is a lie

    def forward(self, x):
        x_1 = torch.roll(x, 1, 3)
        x_1[:,:,:,0] = x_1[:,:,:,1]#.detach()
        x_2 = torch.roll(x, -1, 3)
        x_2[:,:,:,-1] = x_2[:,:,:,-2]#.detach()
        out = self.weight * x_1 + (1. - self.weight) * x_2
        out = self.merge(x, out)
        _ = self.tracker_roll(out)
        return out


class RollModuleVParam(nn.Module):
    def __init__(
        self,
        n_channels : int,
    ) -> None:
        super().__init__()
        
        w = torch.ones(1, n_channels, 1, 1) * 0.5
        self.weight = nn.Parameter(w, True)
        self.weight_limit_min = 0.
        self.weight_limit_max = 1.
        self.tracker_roll = tm.instance_tracker_module(label="Roll V", tracked_module=self, info_code="pass_highlight")
        self.merge = MergeModuleHiddenSymmetric(self.weight)
        #self.merge = MergeModule(n_channels, self.tracker_roll.module_id, self.tracker_roll.module_id) # this is a lie

    def forward(self, x):
        x_1 = torch.roll(x, 1, 2)
        x_1[:,:,0,:] = x_1[:,:,1,:]#.detach()
        x_2 = torch.roll(x, -1, 2)
        x_2[:,:,-1,:] = x_2[:,:,-2,:]#.detach()
        out = self.weight * x_1 + (1. - self.weight) * x_2
        out = self.merge(x, out)
        _ = self.tracker_roll(out)
        return out


class RollGroupParam(nn.Module):
    def __init__(
        self,
        n_channels: int,
    ) -> None:
        super().__init__()
        
        #self.tracker_input = tm.instance_tracker_module(label="Input")
        self.roll_h = RollModuleHParam(n_channels)
        self.roll_v = RollModuleVParam(n_channels)

        
    def forward(self, x):
        #_ = self.tracker_input(x)
        x = self.roll_h(x)
        x = self.roll_v(x)
        return x


class RollGroupFixed(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.tracker_input = tm.instance_tracker_module(label="Input")
        self.tracker_roll = tm.instance_tracker_module(label="Roll")

        
    def forward(self, x):
        _ = self.tracker_input(x)
        # has to match the filter combinations in 3x3 part
        x1 = torch.roll(x[:,0::4], 1, 3) # right
        x1[:,:,:,0] = x1[:,:,:,1].detach()
        x2 = torch.roll(x[:,1::4], 1, 2) # bottom
        x2[:,:,0,:] = x2[:,:,1,:].detach()
        x3 = torch.roll(x[:,2::4], -1, 3) # left
        x3[:,:,:,-1] = x3[:,:,:,-2].detach()
        x4 = torch.roll(x[:,3::4], -1, 2) # top
        x4[:,:,-1,:] = x4[:,:,-2,:].detach()
        x_stacked = zip_tensors([x1, x2, x3, x4])
        _ = self.tracker_roll(x_stacked)
        return x_stacked


class RollGroupRandom(nn.Module):
    def __init__(
        self,
        randomroll : int = 1,
    ) -> None:
        super().__init__()

        self.randomroll = randomroll
        tm.instance_tracker_module_group(label="Roll")

        self.tracker_input = tm.instance_tracker_module(label="Input")
        self.tracker_roll = tm.instance_tracker_module(label="Roll")

        
    def forward(self, x):
        _ = self.tracker_input(x)
        # has to match the filter combinations in 3x3 part
        random_roll_array = [self.randomroll, self.randomroll, -self.randomroll, -self.randomroll]
        random.shuffle(random_roll_array)
        direction_array = [2,2,3,3]
        random.shuffle(direction_array)

        x1 = torch.roll(x[:,0::4], random_roll_array[0], direction_array[0]) # right
        x2 = torch.roll(x[:,1::4], random_roll_array[1], direction_array[1]) # bottom
        x3 = torch.roll(x[:,2::4], random_roll_array[2], direction_array[2]) # left
        x4 = torch.roll(x[:,3::4], random_roll_array[3], direction_array[3]) # top
        x_stacked = zip_tensors([x1, x2, x3, x4])
        _ = self.tracker_roll(x_stacked)
        return x_stacked


class ParamTranslationModule(nn.Module):
    def __init__(
        self,
        n_channels : int = 1,
        k : int = 3,
        dim : int = 2,

    ) -> None:
        super().__init__()

        # Initialize trainable internal weights
        weight = torch.ones(n_channels, 1, 1, 1) * 0.49 # must not be 0.5 because the torch.abs(x) function in forward has no gradient for x=0
        self.weight = nn.Parameter(weight, True)
        self.weight_limit_min = 0.
        self.weight_limit_max = 1.

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

        # Create tracker
        mode = "H" if dim==3 else "V"
        self.tracker_out = tm.instance_tracker_module(label=f"Translation{mode}", tracked_module=self, info_code="pass_highlight")

    def forward(self, x):
        w = - torch.abs(self.weight - self.kernel_positions) / self.radius + 1.
        w = DifferentiableClamp.apply(w, 0., 1.)
        out = F.conv2d(x, w, padding="same", groups=x.shape[1])
        _ = self.tracker_out(out)
        return out


class ParamTranslationGroup(nn.Module):
    def __init__(
        self,
        n_channels : int = 1,
        k : int = 3,

    ) -> None:
        super().__init__()
        
        self.roll_v = ParamTranslationModule(n_channels, k, 2)
        self.roll_h = ParamTranslationModule(n_channels, k, 3)

    def forward(self, x):
        x = self.roll_v(x)
        x = self.roll_h(x)
        return x


class CompareGroup(nn.Module):
    def __init__(
        self,
        n_channels: int,
        clamp_limit_min : float = 0.,
        clamp_limit_max : float = 1.,
    ) -> None:
        super().__init__()

        self.clamp_limit_min = clamp_limit_min
        self.clamp_limit_max = clamp_limit_max
        self.tracker_input = tm.instance_tracker_module(label="Input")
        self.tracker_out = tm.instance_tracker_module(label="AND, OR")
        self.merge = MergeModule(n_channels, self.tracker_input.module_id, self.tracker_out.module_id)


    def forward(self, x):
        _ = self.tracker_input(x)
        x_skip = x
        x_and = x[:,::2] * x[:,1::2]
        x_or = x[:,::2] + x[:,1::2]
        x_or = DifferentiableClamp.apply(x_or, self.clamp_limit_min, self.clamp_limit_max)
        x_stacked = zip_tensors([x_and, x_or])
        _ = self.tracker_out(x_stacked)
        x_merged = self.merge(x_skip, x_stacked)
        return x_merged

def zip_tensors(tensors):
    shape = tensors[0].shape
    tensors = [x.unsqueeze(2) for x in tensors]
    out = torch.cat(tensors, 2)
    out = out.reshape((shape[0], -1, shape[2], shape[3]))
    return out


class PreprocessingModule(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        conv_groups: int = 1,
        avgpool: bool = True,
    ) -> None:
        super().__init__()

        tm.instance_tracker_module_group(label="Preprocessing")

        # Pooling
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) if avgpool else nn.Identity()
        if avgpool:
            self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.tracker_avgpool = tm.instance_tracker_module(label="AvgPool")
        else:

            self.avgpool = nn.Identity()
            self.tracker_avgpool = nn.Identity()

        # Permutation
        group_size = n_channels_in // conv_groups
        self.permutation_module = PermutationModule(torch.arange(n_channels_in).roll(group_size // 2))
        #self.permutation = nn.Parameter(torch.randperm(n_channels_in), False)

        # Copy channels.
        if n_channels_in != n_channels_out:
            self.copymodule = CopyModule(n_channels_in, n_channels_out, interleave=False)
        else:
            self.copymodule = nn.Identity()
            
        # Norm
        self.norm_module = LayerNorm()

    def forward(self, x: Tensor) -> Tensor:
        x = self.avgpool(x)
        _ = self.tracker_avgpool(x)
        x = self.permutation_module(x)
        x = self.copymodule(x)
        x = self.norm_module(x)
        return x

        

class PredefinedFilterBlock(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        conv_groups: int = 1,
        avgpool: bool = True,
        filter_mode: str = "Uneven",
        roll_instead_of_3x3 : bool = False,
        randomroll: int = -1,
    ) -> None:
        super().__init__()

        self.preprocessing = PreprocessingModule(n_channels_in, n_channels_out, conv_groups, avgpool)
        
        # 1x1 Conv
        tm.instance_tracker_module_group(label="1x1 Conv")
        self.mix = MixGroup(n_channels_out, conv_groups)
        
        # Randomroll
        if roll_instead_of_3x3:
            #tm.instance_tracker_module_group(label="Roll")
            self.randomroll = RollGroupRandom(randomroll) if randomroll>0 else nn.Identity()
            self.spatial = RollGroupFixed() if roll_instead_of_3x3 else nn.Identity()
        else:
            tm.instance_tracker_module_group(label="3x3 Conv")
            self.randomroll = RollGroupRandom(randomroll) if randomroll>0 else nn.Identity()
            self.spatial = PredefinedFilterModule3x3Part(
                n_channels_in=n_channels_out,
                filter_mode=ParameterizedFilterMode[filter_mode],
                n_angles=2,
                handcrafted_filters_require_grad=False,
                f=1,
                k=3,
                stride=1,
                activation_layer=nn.LeakyReLU,
            )

        # Merge
        tm.instance_tracker_module_group(label="Merge", precursors=[tm.tracker_module_groups[-3].group_id, tm.tracker_module_groups[-1].group_id])
        self.merge = MergeModule(n_channels_out, self.preprocessing.norm_module.tracker_norm.module_id, tm.module_id, monitor_inputs=True)


    def forward(self, x: Tensor) -> Tensor:
        x = self.preprocessing(x)
        x_skip = x
        x = self.mix(x)
        x = self.randomroll(x)
        x = self.spatial(x)
        x = self.merge(x_skip, x)
        return x


class SpecialTranslationBlock(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        conv_groups: int = 1,
        avgpool: bool = True,
        mode : str = "predefined_filters", # one of predefined_filters, roll, parameterized_translation
        filter_mode: str = "Uneven",
        translation_k : int = 3,
        randomroll: int = -1,
    ) -> None:
        super().__init__()

        self.preprocessing = PreprocessingModule(n_channels_in, n_channels_out, conv_groups, avgpool)
        
        # 1x1 Conv
        tm.instance_tracker_module_group(label="1x1 Conv")
        self.mix = MixGroup(n_channels_out, conv_groups)
        self.merge1 = MergeModule(n_channels_out, self.preprocessing.norm_module.tracker_norm.module_id, tm.module_id, monitor_inputs=False)
        self.randomroll = RollGroupRandom(randomroll) if randomroll>0 else nn.Identity()
        
        if mode == "roll":
            tm.instance_tracker_module_group(label="Roll")
            self.input_tracker = tm.instance_tracker_module(label="Input")
            self.spatial = RollGroupFixed()
        elif mode == "predefined_filters":
            tm.instance_tracker_module_group(label="3x3 Conv")
            self.input_tracker = tm.instance_tracker_module(label="Input")
            self.spatial = PredefinedFilterModule3x3Part(
                n_channels_in=n_channels_out,
                filter_mode=ParameterizedFilterMode[filter_mode],
                n_angles=2,
                handcrafted_filters_require_grad=False,
                f=1,
                k=3,
                stride=1,
                activation_layer=nn.LeakyReLU,
            )
        elif mode == "parameterized_translation":
            tm.instance_tracker_module_group(label="Translation")
            self.input_tracker = tm.instance_tracker_module(label="Input")
            self.spatial = ParamTranslationGroup(n_channels_out, translation_k)

        self.merge2 = MergeModule(n_channels_out, self.input_tracker.module_id, tm.module_id, monitor_inputs=False)

        #tm.instance_tracker_module_group(label="Merge", precursors=[tm.tracker_module_groups[-3].group_id, tm.tracker_module_groups[-1].group_id])


    def forward(self, x: Tensor) -> Tensor:
        x = self.preprocessing(x)
        x_skip = x
        x = self.mix(x)
        x = self.merge1(x_skip, x)
        x = self.randomroll(x)
        x_skip = x
        _ = self.input_tracker(x)
        x = self.spatial(x)
        x = self.merge2(x_skip, x)
        return x


def initialize_weights(modules, init_mode):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            if init_mode == 'uniform':
                nn.init.uniform_(m.weight, m.weight_limit_min, m.weight_limit_max)
            elif init_mode == 'zero':
                nn.init.constant_(m.weight, 0.)
            elif init_mode == 'identity':
                out_channels = m.weight.shape[0]
                group_size = m.weight.shape[1]
                with torch.no_grad():
                    for channel_id in range(out_channels):
                        m.weight[channel_id, channel_id % group_size, 0, 0] = 1.0
        elif isinstance(m, RollModuleHParam) or isinstance(m, RollModuleVParam):
            if init_mode == 'uniform':
                nn.init.uniform_(m.weight, m.weight_limit_min, m.weight_limit_max)
