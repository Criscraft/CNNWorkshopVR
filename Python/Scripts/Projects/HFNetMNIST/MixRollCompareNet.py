from importlib.metadata import requires
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
import numpy as np
import enum

from typing import Type, Any, Callable, Union, List, Tuple, Optional

from torch.nn.modules import module
from Scripts.ActivationTracker import ActivationTracker, TrackerModule, TrackerModuleGroup, reset_ids

# No padding.

TRACKERMODULEGROUPS = []

class MixRollCompareNet(nn.Module):
    def __init__(self,
        n_classes: int = 10,
        start_config: dict = {
            'n_channels_in' : 1,
            'filter_mode' : 'Uneven',
            'n_angles' : 2,
            'handcrafted_filters_require_grad' : False,
            'f' : 4,
            'k' : 3, 
            'stride' : 1,
        },
        blockconfig_list: List[dict] = [
            {'n_channels_in' : 4 if i==0 else 16,
            'n_channels_out' : 16, # n_channels_out % shuffle_conv_groups == 0 and n_channels_out % n_classes == 0 
            'conv_groups' : 16 // 4,
            'avgpool' : True if i in [0, 2] else False,
            } for i in range(4)],
        init_mode: str = 'identity',
        statedict: str = '',
        ):
        super().__init__()

        self.embedded_model = MixRollCompareNet_(
            n_classes=n_classes,
            start_config=start_config,
            blockconfig_list=blockconfig_list, 
            init_mode=init_mode)

        self.tracker_module_groups_info = {group.meta['group_id'] : {key : group.meta[key] for key in ['precursors', 'label']} for group in TRACKERMODULEGROUPS}

        if statedict:
            pretrained_dict = torch.load(statedict, map_location=torch.device('cpu'))
            missing = self.load_state_dict(pretrained_dict, strict=True)
            print('Loading weights from statedict. Missing and unexpected keys:')
            print(missing)
                

    def forward(self, batch):
        if isinstance(batch, dict) and 'data' in batch:
            logits = self.embedded_model(batch['data'])
            out = {'logits' : logits}
            return out
        else:
            return self.embedded_model(batch)


    def forward_features(self, batch, module=None):
        track_modules = ActivationTracker()

        assert isinstance(batch, dict) and 'data' in batch
        output, module_dicts = track_modules.collect_stats(self.embedded_model, batch['data'], module)
        out = {'logits' : output, 'module_dicts' : module_dicts}
        return out

            
    def save(self, statedict_name):
        torch.save(self.state_dict(), statedict_name)


    def regularize(self):
        for m in self.modules():
            if hasattr(m, "weight"):
                if hasattr(m, "weights_min"):
                    m.weight.data = torch.clamp(m.weight.data, m.weights_min, m.weights_max)
                else:
                    m.weight.data = torch.clamp(m.weight.data, -1., 1.)


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
        self.tracker_copymodule = TrackerModule(label="Copy channels", info_code=info_code)

    def forward(self, x: Tensor) -> Tensor:
        if self.interleave:
            out = x.repeat_interleave(self.factor, dim=1)
        else:
            dimensions = [1 for _ in range(x.ndim)]
            dimensions[1] = self.factor
            out =  x.repeat(*dimensions)
        
        _ = self.tracker_copymodule(out)
        
        return out


class NormalizationModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.tracker_norm = TrackerModule(label="ScaleNorm")    

    def forward(self, x: Tensor) -> Tensor:
        dims = tuple(range(x.ndim))[1:]
        minimum = x
        maximum = x
        for dim in dims:
            minimum = minimum.min(dim, keepdims=True)[0]
            maximum = maximum.max(dim, keepdims=True)[0]
        minimum = minimum.detach()
        maximum = maximum.detach()
        out = (x - minimum) / (maximum - minimum + 1e-6)
        _ = self.tracker_norm(out)
        return out
    

class PermutationModule(nn.Module):
    def __init__(self, indices):
        super().__init__()
        self.indices = nn.Parameter(indices, False)
        self.tracker_permutation = TrackerModule(label="Permutation", tracked_module=self)

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


class PredefinedConv(nn.Module):
    def __init__(self, n_channels_in: int, n_channels_out: int, stride: int = 1) -> None:
        super().__init__()

        #self.padding: int = 1
        self.weight: nn.Parameter = None
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.stride = stride
        assert self.n_channels_out >= self.n_channels_in

    
    def forward(self, x: Tensor) -> Tensor:
        n_channels_per_kernel = self.n_channels_out // self.n_kernels

        w_tmp = self.weight.repeat((n_channels_per_kernel, 1, 1, 1)) # this involves copying
        groups = self.n_channels_in

        #x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), "replicate")
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
    def __init__(self, n_channels_in: int, n_channels_out: int, stride: int = 1, k: int = 3, filter_mode: ParameterizedFilterMode = ParameterizedFilterMode.All, n_angles: int = 4, requires_grad=False) -> None:
        super().__init__(n_channels_in, n_channels_out, stride)

        self.padding = k//2
        w = []
        if filter_mode == ParameterizedFilterMode.Uneven or filter_mode == ParameterizedFilterMode.All or filter_mode == ParameterizedFilterMode.UnevenPosOnly:
            w = w + [get_parameterized_filter(k, ParameterizedFilterMode.Uneven, phi) for phi in np.linspace(0, 180, n_angles, endpoint=False)]
        if filter_mode == ParameterizedFilterMode.Even or filter_mode == ParameterizedFilterMode.All or filter_mode == ParameterizedFilterMode.EvenPosOnly:
            w = w + [get_parameterized_filter(k, ParameterizedFilterMode.Even, phi) for phi in np.linspace(0, 180, n_angles, endpoint=False)]
        
        if not filter_mode == ParameterizedFilterMode.UnevenPosOnly and not filter_mode == ParameterizedFilterMode.EvenPosOnly:
            #w = [sign*item for item in w for sign in [-1, 1]]
            w.extend([-w_ for w_ in w])
        
        if filter_mode == ParameterizedFilterMode.Random:
            w = w + [np.random.rand(k, k) * 2. - 1. for _ in range(n_angles)]

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
    ) -> None:
        super().__init__()

        # Copy. This copy module is only decoration. The consecutive module will not be affected by copy.
        self.copymodule = CopyModule(n_channels_in, n_channels_in * f) if f>1 else nn.Identity()
        
        self.predev_conv = PredefinedConvnxn(n_channels_in, n_channels_in * f, stride=stride, k=k, filter_mode=filter_mode, n_angles=n_angles, requires_grad=handcrafted_filters_require_grad)
        self.tracker_predev_conv = TrackerModule(label="3x3 Conv", tracked_module=self.predev_conv)
        
        self.activation_layer = activation_layer(inplace=False)
        if not isinstance(self.activation_layer, nn.Identity):
            self.activation_layer_tracker = TrackerModule(label="ReLU")
        else:
            self.activation_layer_tracker = nn.Identity()


    def forward(self, x: Tensor) -> Tensor:
        _ = self.copymodule(x)
        x = self.predev_conv(x)
        _ = self.tracker_predev_conv(x)
        x = self.activation_layer(x)
        _ = self.activation_layer_tracker(x)
        return x
    

class Conv2d1x1WeightsSumToOne(nn.Module):
    def __init__(
        self,
        n_channels : int, 
        groups : int = 1,
    ) -> None:
        super().__init__()
        
        w = torch.zeros((n_channels, n_channels//groups, 1, 1))
        self.weight = nn.Parameter(w, True)
        self.weights_min = 0.
        self.weights_max = 1.
        self.groups = groups
        self.tracker_conv = TrackerModule(label="1x1 Conv", tracked_module=self)

    def forward(self, x):
        w = self.weight / (self.weight.sum(1, keepdim=True) + 1e-6).detach()
        x = F.conv2d(x, w, groups=self.groups)
        _ = self.tracker_conv(x)
        return x


class NotModule(nn.Module):
    def __init__(
        self,
        n_channels : int,
    ) -> None:
        super().__init__()
        
        w = torch.zeros(1, n_channels, 1, 1)
        self.weight = nn.Parameter(w, True)
        self.weights_min = 0.
        self.weights_max = 1.
        self.tracker_conv = TrackerModule(label="Not", tracked_module=self, info_code="pass_highlight")

    def forward(self, x):
        x = x + self.weight - 2.0*self.weight*x
        _ = self.tracker_conv(x)
        return x


class MixGroup(nn.Module):
    def __init__(
        self,
        n_channels : int,
        conv_groups : int = 1,
    ) -> None:
        super().__init__()

        global TRACKERMODULEGROUPS

        TRACKERMODULEGROUPS.append(TrackerModuleGroup(label="Mix"))

        self.tracker_input = TrackerModule(label="Input")
        self.conv1x1 = Conv2d1x1WeightsSumToOne(n_channels, conv_groups)
        self.not_module = NotModule(n_channels)

    def forward(self, x):
        _ = self.tracker_input(x)
        x = self.conv1x1(x)
        x = self.not_module(x)
        return x


class MergeModule(nn.Module):
    def __init__(
        self,
        n_channels : int,
        module_id_1 : int,
        module_id_2 : int,
    ) -> None:
        super().__init__()
        
        w = torch.zeros(1, n_channels, 1, 1)
        self.weight = nn.Parameter(w, True)
        self.weights_min = 0.
        self.weights_max = 1.
        self.tracker_merge = TrackerModule(label="Merge", tracked_module=self, precursors=[module_id_1, module_id_2])

    def forward(self, x, y):
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


class RollModuleH(nn.Module):
    def __init__(
        self,
        n_channels : int,
    ) -> None:
        super().__init__()
        
        w = torch.ones(1, n_channels, 1, 1) * 0.5
        self.weight = nn.Parameter(w, True)
        self.weights_min = 0.
        self.weights_max = 1.
        self.tracker_roll = TrackerModule(label="Roll H", tracked_module=self, info_code="pass_highlight")
        self.merge = MergeModuleHiddenSymmetric(self.weight)

    def forward(self, x):
        x_1 = torch.roll(x, 1, 3)
        x_1[:,:,:,0] = x_1[:,:,:,1].detach()
        x_2 = torch.roll(x, -1, 3)
        x_2[:,:,:,-1] = x_2[:,:,:,-2].detach()
        out = self.weight * x_1 + (1. - self.weight) * x_2
        out = self.merge(x, out)
        _ = self.tracker_roll(out)
        return out


class RollModuleV(nn.Module):
    def __init__(
        self,
        n_channels : int,
    ) -> None:
        super().__init__()
        
        w = torch.ones(1, n_channels, 1, 1) * 0.5
        self.weight = nn.Parameter(w, True)
        self.weights_min = 0.
        self.weights_max = 1.
        self.tracker_roll = TrackerModule(label="Roll V", tracked_module=self, info_code="pass_highlight")
        self.merge = MergeModuleHiddenSymmetric(self.weight)

    def forward(self, x):
        x_1 = torch.roll(x, 1, 2)
        x_1[:,:,0,:] = x_1[:,:,1,:].detach()
        x_2 = torch.roll(x, -1, 2)
        x_2[:,:,-1,:] = x_2[:,:,-2,:].detach()
        out = self.weight * x_1 + (1. - self.weight) * x_2
        out = self.merge(x, out)
        _ = self.tracker_roll(out)
        return out


class RollGroup(nn.Module):
    def __init__(
        self,
        n_channels: int,
    ) -> None:
        super().__init__()

        global TRACKERMODULEGROUPS

        TRACKERMODULEGROUPS.append(TrackerModuleGroup(label="Roll"))

        self.tracker_input = TrackerModule(label="Input")
        self.roll_h = RollModuleH(n_channels)
        self.roll_v = RollModuleV(n_channels)

        
    def forward(self, x):
        _ = self.tracker_input(x)
        x = self.roll_h(x)
        x = self.roll_v(x)
        return x


class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


class CompareGroup(nn.Module):
    def __init__(
        self,
        n_channels: int,
    ) -> None:
        super().__init__()

        global TRACKERMODULEGROUPS

        TRACKERMODULEGROUPS.append(TrackerModuleGroup(label="Compare"))

        self.tracker_input = TrackerModule(label="Input")
        self.tracker_out = TrackerModule(label="AND, OR")
        self.merge = MergeModule(n_channels, self.tracker_input.module_id, self.tracker_out.module_id)


    def forward(self, x):
        _ = self.tracker_input(x)
        x_skip = x
        x_and = x[:,::2] * x[:,1::2]
        x_or = x[:,::2] + x[:,1::2]
        x_or = DifferentiableClamp.apply(x_or, 0.0, 1.0)
        x_stacked = self.zip_tensors(x_and, x_or)
        _ = self.tracker_out(x_stacked)
        x_merged = self.merge(x_skip, x_stacked)
        return x_merged

    def zip_tensors(self, x1, x2):
        shape = x1.shape
        x1 = x1.unsqueeze(2)
        x2 = x2.unsqueeze(2)
        out = torch.cat([x1, x2], 2)
        out = out.reshape((shape[0], -1, shape[2], shape[3]))
        return out


class MixRollCompareBlock(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        conv_groups: int = 1,
        avgpool: bool = True,
    ) -> None:
        super().__init__()

        global TRACKERMODULEGROUPS

        TRACKERMODULEGROUPS.append(TrackerModuleGroup(label="Preprocessing"))

        # Pooling
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) if avgpool else nn.Identity()
        if avgpool:
            self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            self.tracker_avgpool = TrackerModule(label="AvgPool")
        else:
            self.avgpool = nn.Identity()
            self.tracker_avgpool = nn.Identity()

        # Norm
        self.norm_module = NormalizationModule()

        # Permutation
        group_size = n_channels_in // conv_groups
        self.permutation_module = PermutationModule(torch.arange(n_channels_in).roll(group_size // 2))
        #self.permutation = nn.Parameter(torch.randperm(n_channels_in), False)

        # Copy channels.
        if n_channels_in != n_channels_out:
            self.copymodule = CopyModule(n_channels_in, n_channels_out, interleave=False)
        else:
            self.copymodule = nn.Identity()

        # Mix Roll and Compare
        self.mix = MixGroup(n_channels_out, conv_groups)
        self.roll = RollGroup(n_channels_out)
        self.compare = CompareGroup(n_channels_out)


    def forward(self, x: Tensor) -> Tensor:
        x = self.avgpool(x)
        _ = self.tracker_avgpool(x)
        x = self.norm_module(x)
        x = self.permutation_module(x)
        x = self.copymodule(x)
        x = self.mix(x)
        x = self.roll(x)
        x = self.compare(x)
        return x
    

class MixRollCompareNet_(nn.Module):

    def __init__(
        self,
        n_classes: int = 10,
        start_config: dict = {
            'n_channels_in' : 1,
            'filter_mode' : 'Uneven',
            'n_angles' : 2,
            'handcrafted_filters_require_grad' : False,
            'f' : 4,
            'k' : 3, 
            'stride' : 1,
        },
        blockconfig_list: List[dict] = [
            {'n_channels_in' : 4 if i==0 else 16,
            'n_channels_out' : 16, # n_channels_out % shuffle_conv_groups == 0 and n_channels_out % n_classes == 0 
            'conv_groups' : 16 // 4,
            'avgpool' : True if i in [0, 2] else False,
            } for i in range(4)],
        init_mode: str = 'identity',
    ) -> None:
        super().__init__()
        
        global TRACKERMODULEGROUPS
        # reset, because a second network instance could change the globals
        TRACKERMODULEGROUPS = []
        # Reset the id system of the activaion tracker. This is necessary when multiple networks are instanced.
        reset_ids()

        # Input
        TRACKERMODULEGROUPS.append(TrackerModuleGroup(label="Input", precursors=[]))
        self.tracker_input = TrackerModule(label="Input", precursors=[])

        TRACKERMODULEGROUPS.append(TrackerModuleGroup(label="First Convolution"))
        self.conv1 = PredefinedFilterModule3x3Part(
            n_channels_in=start_config['n_channels_in'],
            filter_mode=ParameterizedFilterMode[start_config['filter_mode']],
            n_angles=start_config['n_angles'],
            handcrafted_filters_require_grad=start_config['handcrafted_filters_require_grad'],
            f=start_config['f'],
            k=start_config['k'],
            stride=start_config['stride'],
            activation_layer=nn.ReLU,
        )

        # Blocks
        blocks = [
            MixRollCompareBlock(
                n_channels_in=config['n_channels_in'],
                n_channels_out=config['n_channels_out'],
                conv_groups=config['conv_groups'],
                avgpool=config['avgpool'],
            ) for config in blockconfig_list]
        self.blocks = nn.Sequential(*blocks)

        # AdaptiveAvgPool
        self.adaptiveavgpool = nn.AdaptiveAvgPool2d((1, 1))

        TRACKERMODULEGROUPS.append(TrackerModuleGroup(label="AvgPool"))
        self.tracker_adaptiveavgpool = TrackerModule(label="AvgPool")

        # Classifier
        self.classifier = nn.Conv2d(blockconfig_list[-1]['n_channels_out'], n_classes, kernel_size=1, stride=1, bias=False)
        # For the regularization we have to give the classifier weights_min and weights_max
        self.classifier.weights_min = -1.0
        self.classifier.weights_max = 1.0
        TRACKERMODULEGROUPS.append(TrackerModuleGroup(label="Classifier"))
        self.tracker_classifier = TrackerModule(label="Classifier", tracked_module=self.classifier, channel_labels="classes")
        self.tracker_classifier_softmax = TrackerModule(label="Class Probabilities", channel_labels="classes")
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, Conv2d1x1WeightsSumToOne):
                if init_mode == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=nn.ReLU)
                elif init_mode == 'zero':
                    nn.init.constant_(m.weight, 0.)
                elif init_mode == 'identity':
                    out_channels = m.weight.shape[0]
                    group_size = m.weight.shape[1]
                    with torch.no_grad():
                        for channel_id in range(out_channels):
                            m.weight[channel_id, channel_id % group_size, 0, 0] = 1.0
        

    def forward(self, x: Tensor) -> Tensor:
        _ = self.tracker_input(x)
        x = self.conv1(x)
        x = self.blocks(x)

        x = self.adaptiveavgpool(x)
        _ = self.tracker_adaptiveavgpool(x)
        x = self.classifier(x)
        _ = self.tracker_classifier(x)
        _ = self.tracker_classifier_softmax(F.softmax(x, 1))
        x = torch.flatten(x, 1)

        return x