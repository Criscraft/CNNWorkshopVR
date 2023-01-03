import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import enum
import random

from TrackingModules import TrackerModuleProvider

tm = TrackerModuleProvider()

class WeightClampingModule(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.params_to_clamp = []
        self.params_to_cycle = []

    def register_param_to_clamp(self, param, minimum, maximum):
        self.params_to_clamp.append({'param' : param, 'min' : minimum, 'max' : maximum})

    def register_param_to_cycle(self, param, minimum, maximum):
        self.params_to_clamp.append({'param' : param, 'min' : minimum, 'max' : maximum})

    def regularize_params(self):

        for item in self.params_to_clamp:
            item['param'].data = torch.clamp(item['param'].data, item['min'], item['max'])

        for item in self.params_to_cycle:
            item['param'].data = item['param'].data - item['min']
            item['param'].data = item['param'].data % (item['max'] - item['min'])
            item['param'].data = item['param'].data + item['min']

    def intitialize_weights_uniform(self):
        for item in self.params_to_clamp:
            nn.init.uniform_(item['param'], item['min'], item['max'])

    def intitialize_weights_zero(self):
        for item in self.params_to_clamp:
            nn.init.constant_(item['param'], 0.)

    def intitialize_weights_identity(self):
        for item in self.params_to_clamp:
            nn.init.constant_(item['param'], 1.)


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


class TrackedConv1x1(WeightClampingModule):
    def __init__(
        self, n_channels_in, n_channels_out, conv_groups, clamp_min=-1., clamp_max=1.,
    ) -> None:
        super().__init__()

        self.conv1x1 = nn.Conv2d(n_channels_in, n_channels_out, 1, 1, 0, groups=conv_groups, bias=False)
        self.register_param_to_clamp(self.conv1x1.weight, clamp_min, clamp_max)
        self.create_trackers(n_channels_in, n_channels_out, conv_groups)

    def create_trackers(self, in_channels, out_channels, conv_groups, clamp_min=-1., clamp_max=1.):
        self.tracker_out = tm.instance_tracker_module(label="1x1 Conv", draw_edges=True, ignore_highlight=False)
        self.tracker_out.register_data("grouped_conv_weight", self.conv1x1.weight)
        self.tracker_out.register_data("grouped_conv_weight_range", [clamp_min, clamp_max])
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
    def __init__(self) -> None:
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

    # Called in init by subclass
    def create_trackers(self):
        self.tracker_out = tm.instance_tracker_module(label="PFModule")
        self.tracker_out.register_data("PFModule_kernels", self.weight)
    
    def forward(self, x: Tensor) -> Tensor:
        n_channels_per_kernel = self.n_channels_out // self.n_kernels

        w_tmp = self.weight.repeat((n_channels_per_kernel, 1, 1, 1)) # this involves copying
        groups = self.n_channels_in

        if self.padding:
            x = F.pad(x, (1,1,1,1), "replicate")
        out = F.conv2d(x, w_tmp, None, self.stride, groups=groups)

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


class Conv1x1AndReLUModule(WeightClampingModule):
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


class SparseConv2D(WeightClampingModule):
    def __init__(
        self,
        n_channels : int,
        conv_groups : int,
        n_selectors : int, 
        selector_radius : int, # the nth channel before or after the current channel will contribute when n<selector_radius
    ) -> None:
        super().__init__()

        self.n_selectors = n_selectors
        
        # Initialize trainable internal weights
        # Shape of weight_selection: [n_selectors, n_channels_out, 1 (group_size), filter height, filter width]
        weight_selection = torch.ones(self.n_selectors, n_channels, 1, 1, 1) * 0.49 # must not be 0.5 because the torch.abs(x) function in forward has no gradient for x=0
        self.weight_selection = nn.Parameter(weight_selection, True)
        nn.init.uniform_(self.weight_selection, 0., 1.)
        self.register_param_to_clamp(self.weight_selection, 0., 1.)
        # Shape of weight_group: [n_selectors, 1 (batchsize), n_channels, tensor height, tensor width]
        weight_group = torch.zeros(self.n_selectors, 1, n_channels, 1, 1)
        self.weight_group = nn.Parameter(weight_group, True)
        nn.init.uniform_(self.weight_group, -1., 1.)
        self.register_param_to_cycle(self.weight_group, -1., 1.)

        # Initialize helpers
        self.conv_groups = conv_groups
        group_size = n_channels // conv_groups
        # shape: [1 (n_selectors), 1 (n_channels_out), group_size, 1 (filter height), 1 (filter width)]
        kernel_positions = torch.zeros(1, 1, group_size)
        for i in range(group_size):
                kernel_positions[:,:,i] = i/(group_size-1.)
        kernel_positions = kernel_positions.unsqueeze(3).unsqueeze(4)
        self.kernel_positions = nn.Parameter(kernel_positions, False)
        self.radius = selector_radius / (group_size-1.)

        self.create_trackers(group_size, n_channels)
        self.relu = TrackedLeakyReLU()

    def create_trackers(self, group_size, n_channels):
        self.tracker = tm.instance_tracker_module(label="Sparse Conv", draw_edges=True, ignore_highlight=False)
        self.tracker.register_data("sparse_conv_weight_selection", self.weight_selection)
        self.tracker.register_data("sparse_conv_weight_selection_range", [0., 1.])
        self.tracker.register_data("sparse_conv_weight_group", self.weight_group)
        self.tracker.register_data("sparse_conv_weight_group_range", [-1., 1.])
        self.tracker.register_data("radius", self.radius)
        self.tracker.register_data("group_size", group_size)
        in_channels_per_group = n_channels // self.conv_groups
        out_channels_per_group = n_channels // self.conv_groups
        input_mapping = []
        for out_channel in range(n_channels):
            group = out_channel // out_channels_per_group
            input_mapping.append(list(range(group*in_channels_per_group, (group+1)*in_channels_per_group)))
        self.tracker.register_data("input_mapping", input_mapping)
        
        # input_mapping = []
        # in_channels_per_group = n_channels // conv_groups
        # out_channels_per_group = n_channels // conv_groups

        # distances = torch.abs(self.weight_selection - self.kernel_positions)
        # # make distances circular
        # distances = torch.minimum(distances, 1. - distances)
        # distances = distances.flatten(2)

        # for selector in range(n_selectors):
        #     input_mapping_selector = []
        #     for out_channel in range(n_channels):
        #         group = out_channel // out_channels_per_group
        #         input_mapping_group = list(range(group*in_channels_per_group, (group+1)*in_channels_per_group))
        #         selected_index = torch.argmin(distances[selector, out_channel].flatten()).cpu().item()
        #         selected_indices = list(range(selected_index-selector_diameter//2, selected_index+selector_diameter//2+1))
        #         input_mapping_selector.append([input_mapping_group[i] for i in selected_indices])
        #     input_mapping.append(input_mapping_selector)
        # self.tracker_out.register_data("input_mapping", input_mapping)

    def forward(self, x):
        distances = torch.abs(self.weight_selection - self.kernel_positions)
        # make distances circular
        distances = torch.minimum(distances, 1. - distances)
        # Create convolution weights
        # When a distance exeeds the radius, the corresponding channel will not contribute.
        w = 1. - distances / self.radius
        w = DifferentiableClamp.apply(w, 0., 1.)
        tensor_list = []
        for selector in range(self.n_selectors):
            y = F.conv2d(x, w[selector], groups=self.conv_groups)
            y = y * self.weight_group[selector]
            tensor_list.append(y)
        x = torch.sum(torch.stack(tensor_list), 0)
        _ = self.tracker(x)
        x = self.relu(x)
        return x


class BlendModule(WeightClampingModule):
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
        self.tracker_out.register_data("blend_weight_range", [0., 1.])

    def forward(self, x, y):
        _ = self.input_tracker_1(x)
        _ = self.input_tracker_2(y)
        out = self.weight * y + (1.0 - self.weight) * x
        _ = self.tracker_out(out)
        return out


# class RollGroupFixed(nn.Module):
#     def __init__(
#         self,
#     ) -> None:
#         super().__init__()

#         self.tracker_input = tm.instance_tracker_module(label="Input")
#         self.tracker_roll = tm.instance_tracker_module(label="Roll")

        
#     def forward(self, x):
#         _ = self.tracker_input(x)
#         # has to match the filter combinations in 3x3 part
#         x1 = torch.roll(x[:,0::4], 1, 3) # right
#         x1[:,:,:,0] = x1[:,:,:,1].detach()
#         x2 = torch.roll(x[:,1::4], 1, 2) # bottom
#         x2[:,:,0,:] = x2[:,:,1,:].detach()
#         x3 = torch.roll(x[:,2::4], -1, 3) # left
#         x3[:,:,:,-1] = x3[:,:,:,-2].detach()
#         x4 = torch.roll(x[:,3::4], -1, 2) # top
#         x4[:,:,-1,:] = x4[:,:,-2,:].detach()
#         x_stacked = zip_tensors([x1, x2, x3, x4])
#         _ = self.tracker_roll(x_stacked)
#         return x_stacked

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


class ParamTranslationModule(WeightClampingModule):
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
        self.tracker_out.register_data("weight_per_channel_range", [0., 1.])

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
            data [0::4] = item['max']
            data [1::4] = 0.49
            data [2::4] = item['min']
            data [3::2] = 0.49
            data = data.reshape(shape)
            item['param'].data = data


    def intitialize_weights_antialternating(self):
        for item in self.params_to_clamp:
            data = item['param'].data
            shape = data.shape
            data = data.flatten()
            data [0::4] = 0.49
            data [1::4] = item['max']
            data [2::4] = 0.49
            data [3::2] = item['min']
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


# class CompareGroup(nn.Module):
#     def __init__(
#         self,
#         n_channels: int,
#         clamp_limit_min : float = 0.,
#         clamp_limit_max : float = 1.,
#     ) -> None:
#         super().__init__()

#         self.clamp_limit_min = clamp_limit_min
#         self.clamp_limit_max = clamp_limit_max
#         self.tracker_input = tm.instance_tracker_module(label="Input")
#         self.tracker_out = tm.instance_tracker_module(label="AND, OR")
#         self.merge = MergeModule(n_channels, self.tracker_input.module_id, self.tracker_out.module_id)


#     def forward(self, x):
#         _ = self.tracker_input(x)
#         x_skip = x
#         x_and = x[:,::2] * x[:,1::2]
#         x_or = x[:,::2] + x[:,1::2]
#         x_or = DifferentiableClamp.apply(x_or, self.clamp_limit_min, self.clamp_limit_max)
#         x_stacked = zip_tensors([x_and, x_or])
#         _ = self.tracker_out(x_stacked)
#         x_merged = self.merge(x_skip, x_stacked)
#         return x_merged

# def zip_tensors(tensors):
#     shape = tensors[0].shape
#     tensors = [x.unsqueeze(2) for x in tensors]
#     out = torch.cat(tensors, 2)
#     out = out.reshape((shape[0], -1, shape[2], shape[3]))
#     return out


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

        # Input tracker
        self.tracker_in = tm.instance_tracker_module(label="Input")

        # Pooling
        self.avgpool = TrackedAvgPool2d(kernel_size=2, stride=2, padding=0) if avgpool else nn.Identity()

        # Permutation
        group_size = n_channels_in // conv_groups
        self.permutation_module = PermutationModule(torch.arange(n_channels_in).roll(group_size // 2))
        #self.permutation = nn.Parameter(torch.randperm(n_channels_in), False)

        # Copy channels.
        if n_channels_in != n_channels_out:
            self.copymodule = CopyModuleNonInterleave(n_channels_in, n_channels_out)
        else:
            self.copymodule = nn.Identity()
            
        # Norm
        self.norm_module = LayerNorm()

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
        n_channels_out: int,
        conv_groups: int = 1,
        avgpool: bool = True,
        conv_mode : str = "default", # one of default, sparse
        sparse_conv_selector_radius : int = 1,
        spatial_mode : str = "predefined_filters", # one of predefined_filters, parameterized_translation
        spatial_blending : bool = True,
        spatial_requires_grad : bool = True,
        filter_mode: str = "Uneven",
        translation_k : int = 3,
        randomroll: int = -1,
    ) -> None:
        super().__init__()

        self.preprocessing = PreprocessingModule(n_channels_in, n_channels_out, conv_groups, avgpool)
        
        # 1x1 Conv
        tm.instance_tracker_module_group(label="1x1 Conv")
        self.tracker_input_conv = tm.instance_tracker_module(label="Input")
        if conv_mode == "default":
            self.conv1x1 = Conv1x1AndReLUModule(n_channels_out, conv_groups)
        elif conv_mode == "sparse":
            self.conv1x1 = SparseConv2D(n_channels_out, conv_groups, 2, sparse_conv_selector_radius)
        self.blend1 = BlendModule(n_channels_out, self.tracker_input_conv.module_id, tm.module_id, monitor_inputs=False)
        self.randomroll = RandomRoll(randomroll) if randomroll>0 else nn.Identity()
        
        # if mode == "roll":
        #     tm.instance_tracker_module_group(label="Roll")
        #     self.tracker = tm.instance_tracker_module(label="Input")
        #     self.spatial = RollGroupFixed()
        if spatial_mode == "predefined_filters":
            tm.instance_tracker_module_group(label="3x3 Conv")
            self.tracker_input_spatial = tm.instance_tracker_module(label="Input")
            self.spatial = PredefinedFilterModule3x3Part(
                n_channels_in=n_channels_out,
                filter_mode=ParameterizedFilterMode[filter_mode],
                n_angles=2,
                handcrafted_filters_require_grad=spatial_requires_grad,
                f=1,
                k=3,
                stride=1,
            )
        elif spatial_mode == "parameterized_translation":
            tm.instance_tracker_module_group(label="Translation")
            self.tracker_input_spatial = tm.instance_tracker_module(label="Input")
            self.spatial = ParamTranslationGroup(n_channels_out, translation_k, spatial_requires_grad)

        if spatial_blending:
            self.blend2 = BlendModule(n_channels_out, self.tracker_input_spatial.module_id, tm.module_id, monitor_inputs=False)
        else: 
            self.blend2 = nn.Identity()
        #tm.instance_tracker_module_group(label="Merge", precursors=[tm.tracker_module_groups[-3].group_id, tm.tracker_module_groups[-1].group_id])


    def forward(self, x: Tensor) -> Tensor:
        x = self.preprocessing(x)
        x_skip = x
        _ = self.tracker_input_conv(x)
        x = self.conv1x1(x)
        x = self.blend1(x_skip, x)
        x = self.randomroll(x)
        x_skip = x
        _ = self.tracker_input_spatial(x)
        x = self.spatial(x)
        x = self.blend2(x_skip, x)
        return x


def initialize_weights(modules, init_mode):
    for m in modules:
        if isinstance(m, WeightClampingModule):
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