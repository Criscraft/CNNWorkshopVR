from importlib.metadata import requires
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import enum

from typing import Type, Any, Callable, Union, List, Tuple, Optional

from torch.nn.modules import module
from Scripts.ActivationTracker import ActivationTracker, TrackerModule, TrackerModuleGroup, reset_ids

# More frequent skip connections. The 3x3 and 1x1 convs switch places.

TRACKERMODULEGROUPS = []

class PFNetSimple(nn.Module):
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
            {'n_channels_in' : 20,
            'n_channels_out' : 20, # n_channels_out % shuffle_conv_groups == 0
            'filter_mode' : 'Uneven',
            'n_angles' : 2,
            'f' : 1,
            'k' : 3, 
            'handcrafted_filters_require_grad' : False,
            'shuffle_conv_groups' : 20 // 4,
            'avgpool' : True if i>0 else False,
            } for i in range(3)],
        avgpool_after_firstlayer: bool = False,
        init_mode: str = 'kaiming_normal',
        statedict: str = '',
        ):
        super().__init__()

        self.embedded_model = PFNet_(
            n_classes=n_classes,
            start_config=start_config,
            blockconfig_list=blockconfig_list, 
            avgpool_after_firstlayer=avgpool_after_firstlayer,
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


class LayerNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        dims = tuple(range(x.ndim))[1:]
        mean = x.mean(dim=dims, keepdims=True)
        std = x.std(dim=dims, keepdims=True)
        return (x - mean) / (std + 1e-6)
    

class RewireModule(nn.Module):
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

        self.padding: int = 1
        self.w: nn.Parameter = None
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        self.stride = stride
        #self.permutation = nn.Parameter(torch.randperm(n_channels_out), False)
        assert self.n_channels_out >= self.n_channels_in

    
    def forward(self, x: Tensor) -> Tensor:
        #assert self.n_channels_out % self.n_kernels == 0
        n_channels_per_kernel = self.n_channels_out // self.n_kernels

        w_tmp = self.w.repeat((n_channels_per_kernel, 1, 1, 1)) # this involves copying
        #w_tmp = w_tmp[self.permutation]
        # w_tmp has shape (out_channels,in_channels​/groups,kH,kW)
        groups = self.n_channels_in

        out = F.conv2d(x, w_tmp, None, self.stride, self.padding, groups=groups)
        #print(f"multadds {x.shape[2]*x.shape[3]*self.n_channels_out*self.w.shape[1]*self.w.shape[2]}")
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


class SmoothConv(nn.Module):
    def __init__(
        self,
        k: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.padding = k//2
        self.stride = stride

        w = [get_parameterized_filter(k, ParameterizedFilterMode.Smooth)]
        w = torch.FloatTensor(np.array(w))
        w = w.unsqueeze(1)
        self.w = nn.Parameter(w, False)


    def forward(self, x: Tensor) -> Tensor:
        n_channels_in = x.shape[1]
        w_tmp = self.w.repeat((n_channels_in, 1, 1, 1))
        out = F.conv2d(x, w_tmp, None, self.stride, self.padding, groups=n_channels_in)
        return out
    

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
            w = [sign*item for item in w for sign in [-1, 1]]
        
        if filter_mode == ParameterizedFilterMode.Random:
            w = w + [np.random.rand(k, k) * 2. - 1. for _ in range(n_angles)]

        self.n_kernels = len(w)
        w = torch.FloatTensor(np.array(w))
        w = w.unsqueeze(1)
        self.w = nn.Parameter(w, requires_grad)


class PredefinedFilterModule1x1Part(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        conv: module = nn.Conv2d,
        shuffle_conv_groups: int = 1,
        activation_layer : nn.Module = nn.ReLU,
    ) -> None:
        super().__init__()
        
        # Copy. This copy module is only decoration. The consecutive module will not be affected by copy.
        if n_channels_in < n_channels_out:
            self.copymodule = CopyModule(n_channels_in, n_channels_out, interleave=False)
        else:
            self.copymodule = nn.Identity()

        self.conv1x1 = conv(n_channels_in, n_channels_out, kernel_size=1, stride=1, bias=False, groups=shuffle_conv_groups)

        self.tracker_conv1x1 = TrackerModule(label="1x1 Conv", tracked_module=self.conv1x1)

        self.activation_layer = activation_layer(inplace=False)
        if not isinstance(self.activation_layer, nn.Identity):
            self.activation_layer_tracker = TrackerModule(label="ReLU")
        else:
            self.activation_layer_tracker = nn.Identity()


    def forward(self, x: Tensor) -> Tensor:
        _ = self.copymodule(x)
        x = self.conv1x1(x)
        _ = self.tracker_conv1x1(x)
        x = self.activation_layer(x)
        _ = self.activation_layer_tracker(x)
        return x
    

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

        self.norm = LayerNorm()
        self.tracker_norm = TrackerModule(label="LayerNorm")


    def forward(self, x: Tensor) -> Tensor:
        _ = self.copymodule(x)
        x = self.predev_conv(x)
        _ = self.tracker_predev_conv(x)
        x = self.activation_layer(x)
        _ = self.activation_layer_tracker(x)
        x = self.norm(x)
        _ = self.tracker_norm(x)
        return x
    

class PredefinedFilterModule(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        filter_mode: ParameterizedFilterMode,
        n_angles: int,
        f: int = 1,
        k: int = 3,
        stride: int = 1,
        activation_layer : nn.Module = nn.ReLU,
        handcrafted_filters_require_grad: bool = False,
        conv: module = nn.Conv2d,
        shuffle_conv_groups: int = 1,
        activation_for_3x3: bool = False
    ) -> None:
        super().__init__()

        global TRACKERMODULEGROUPS

        TRACKERMODULEGROUPS.append(TrackerModuleGroup(label="PFModule"))

        self.tracker_input = TrackerModule(label="Input")
        self.predefined_filter_module_1x1_part = PredefinedFilterModule1x1Part(
            n_channels_in=n_channels_in,
            n_channels_out=n_channels_out,
            conv=conv,
            shuffle_conv_groups=shuffle_conv_groups,
            activation_layer=activation_layer,
        )
        self.predefined_filter_module_3x3_part = PredefinedFilterModule3x3Part(
            n_channels_in=n_channels_out,
            filter_mode=filter_mode,
            n_angles=n_angles,
            handcrafted_filters_require_grad=handcrafted_filters_require_grad,
            f=f,
            k=k,
            stride=stride,
            activation_layer=activation_layer if activation_for_3x3 else nn.Identity,
        )

        
    def forward(self, x):
        _ = self.tracker_input(x)
        x = self.predefined_filter_module_1x1_part(x)
        x = self.predefined_filter_module_3x3_part(x)
        return x


class ResidualModule(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        filter_mode: ParameterizedFilterMode,
        n_angles: int,
        f: int = 1,
        k: int = 3,
        stride: int = 1,
        activation_layer : nn.Module = nn.ReLU,
        handcrafted_filters_require_grad: bool = False,
        conv: module = nn.Conv2d,
        shuffle_conv_groups: int = 1,
        avgpool: bool = True,
    ) -> None:
        super().__init__()

        global TRACKERMODULEGROUPS

        # Pooling
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) if avgpool else nn.Identity()
        if avgpool:
            self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            TRACKERMODULEGROUPS.append(TrackerModuleGroup(label="AvgPool"))
            self.tracker_avgpool = TrackerModule(label="AvgPool")
        else:
            self.avgpool = nn.Identity()
            self.tracker_avgpool = nn.Identity()

        # Rewiring
        TRACKERMODULEGROUPS.append(TrackerModuleGroup(label="Channel Permutation"))
        summand_group_id = TRACKERMODULEGROUPS[-1].group_id
        group_size = n_channels_in // shuffle_conv_groups
        self.rewire_module = RewireModule(torch.arange(n_channels_in).roll(group_size // 2))
        summand_module_id = self.rewire_module.tracker_permutation.module_id
        #self.permutation = nn.Parameter(torch.randperm(n_channels_in), False)

        # Copy channels.
        if n_channels_in != n_channels_out:
            self.copymodule = CopyModule(n_channels_in, n_channels_out, interleave=True)
            summand_group_id = TRACKERMODULEGROUPS[-1].group_id
            summand_module_id = self.copymodule.tracker_copymodule.module_id
        else:
            self.copymodule = nn.Identity()

        # HFModule
        self.pfmodule = PredefinedFilterModule(
            n_channels_in=n_channels_in,
            n_channels_out=n_channels_out,
            filter_mode=filter_mode,
            n_angles=n_angles,
            f=f,
            k=k,
            stride=stride,
            activation_layer=activation_layer,
            handcrafted_filters_require_grad=handcrafted_filters_require_grad,
            conv=conv,
            shuffle_conv_groups=shuffle_conv_groups,
            activation_for_3x3=False
            )
        pf_module_id = self.pfmodule.predefined_filter_module_3x3_part.tracker_norm.module_id

        # Skip
        TRACKERMODULEGROUPS.append(TrackerModuleGroup(precursors=[summand_group_id], label="Skip"))
        self.skip = conv(n_channels_out, n_channels_out, kernel_size=1, stride=1, bias=False, groups=n_channels_out)
        self.tracker_skip = TrackerModule(label="1x1 Conv", tracked_module=self.skip, precursors=[summand_module_id])
        skip_module_id = self.tracker_skip.module_id

        # Sum
        TRACKERMODULEGROUPS.append(TrackerModuleGroup(precursors=[-1, -2], label="Sum"))

        self.tracker_sum_summand1 = TrackerModule(label="Summand 1", precursors=[pf_module_id])
        self.tracker_sum_summand2 = TrackerModule(label="Summand 2", precursors=[skip_module_id])
        self.tracker_sum = TrackerModule(label="Sum", precursors=[-1, -2])

        # ReLU and norm
        TRACKERMODULEGROUPS.append(TrackerModuleGroup(label="Activation and Norm"))

        self.activation_layer = activation_layer(inplace=False)
        self.tracker_activation_layer = TrackerModule(label="ReLU")

        self.norm = LayerNorm()
        self.tracker_norm = TrackerModule(label="LayerNorm")       


    def forward(self, x: Tensor) -> Tensor:
        x = self.avgpool(x)
        _ = self.tracker_avgpool(x)
        x = self.rewire_module(x)
        _ = self.copymodule(x)

        out = self.pfmodule(x)

        x = self.skip(x)
        _ = self.tracker_skip(x)
        
        _ = self.tracker_sum_summand1(out)
        _ = self.tracker_sum_summand2(x)
        out = out + x
        _ = self.tracker_sum(out)

        out = self.activation_layer(out)
        _ = self.tracker_activation_layer(out)
        out = self.norm(out)
        _ = self.tracker_norm(out)

        return out
    

class PFNet_(nn.Module):

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
            {'n_channels_in' : 20,
            'n_channels_out' : 20, # n_channels_out % shuffle_conv_groups == 0
            'filter_mode' : 'Uneven',
            'n_angles' : 2,
            'f' : 1,
            'k' : 3, 
            'handcrafted_filters_require_grad' : False,
            'shuffle_conv_groups' : 20 // 4,
            'avgpool' : True if i>0 else False,
            } for i in range(3)],
        activation: str = 'relu',
        avgpool_after_firstlayer: bool = True,
        init_mode: str = 'kaiming_normal',
    ) -> None:
        super().__init__()

        if activation == 'relu':
            activation_layer = nn.ReLU
        elif activation == 'leaky_relu':
            activation_layer = nn.LeakyReLU
        self._activation_layer = activation_layer

        global TRACKERMODULEGROUPS
        # reset, because a second network instance could change the globals
        TRACKERMODULEGROUPS = []
        # Reset the id system of the activaion tracker. This is necessary when multiple networks are instanced.
        reset_ids()

        # Input
        TRACKERMODULEGROUPS.append(TrackerModuleGroup(label="Input", precursors=[]))
        self.tracker_input_1 = TrackerModule(label="Input", precursors=[])

        TRACKERMODULEGROUPS.append(TrackerModuleGroup(label="First Convolution"))
        self.conv1 = PredefinedFilterModule3x3Part(
            n_channels_in=start_config['n_channels_in'],
            filter_mode=ParameterizedFilterMode[start_config['filter_mode']],
            n_angles=start_config['n_angles'],
            handcrafted_filters_require_grad=start_config['handcrafted_filters_require_grad'],
            f=start_config['f'],
            k=start_config['k'],
            stride=start_config['stride'],
            activation_layer=self._activation_layer,
        )
        
        # Pooling
        TRACKERMODULEGROUPS.append(TrackerModuleGroup(label="AvgPool"))
        self.avgpool_after_first_layer = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) if avgpool_after_firstlayer else nn.Identity()
        self.tracker_avgpool_after_first_layer = TrackerModule(label="AvgPool")

        # Copy
        TRACKERMODULEGROUPS.append(TrackerModuleGroup(label="Copy Channels"))
        if start_config['n_channels_in'] * start_config['f'] < blockconfig_list[0]['n_channels_in']:
            self.copymodule = CopyModule(start_config['n_channels_in'] * start_config['f'], blockconfig_list[0]['n_channels_in'], interleave=False)
        else:
            self.copymodule = nn.Identity()

        # Blocks
        blocks = [
            ResidualModule(
                n_channels_in=config['n_channels_in'],
                n_channels_out=config['n_channels_out'],
                filter_mode=ParameterizedFilterMode[config['filter_mode']],
                n_angles=config['n_angles'],
                f=config['f'],
                k=config['k'],
                activation_layer=self._activation_layer,
                handcrafted_filters_require_grad=config['handcrafted_filters_require_grad'],
                shuffle_conv_groups=config['shuffle_conv_groups'],
                avgpool=config['avgpool'],
            ) for config in blockconfig_list]
        self.blocks = nn.Sequential(*blocks)

        # AdaptiveAvgPool
        self.adaptiveavgpool = nn.AdaptiveAvgPool2d((1, 1))

        TRACKERMODULEGROUPS.append(TrackerModuleGroup(label="AvgPool"))
        self.tracker_adaptiveavgpool = TrackerModule(label="AvgPool")

        # Classifier
        self.classifier = nn.Conv2d(blockconfig_list[-1]['n_channels_out'], n_classes, kernel_size=1, stride=1, bias=False)
        TRACKERMODULEGROUPS.append(TrackerModuleGroup(label="Classifier"))
        self.tracker_classifier = TrackerModule(label="Classifier", tracked_module=self.classifier, channel_labels="classes")
        self.tracker_classifier_softmax = TrackerModule(label="Class Probabilities", channel_labels="classes")
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and hasattr(m, 'kernel_size') and m.kernel_size[0] == 1:
                if init_mode == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
                elif init_mode == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity=activation)
                elif init_mode == 'sparse':
                    nn.init.sparse_(m.weight, sparsity=0.1, std=0.01)
                elif init_mode == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=1)
                elif init_mode == 'zero':
                    nn.init.constant_(m.weight, 0.)
        

    def forward(self, x: Tensor) -> Tensor:
        _ = self.tracker_input_1(x)
        x = self.conv1(x)
        x = self.avgpool_after_first_layer(x)
        _ = self.tracker_avgpool_after_first_layer(x)

        x = self.copymodule(x)
        x = self.blocks(x)

        x = self.adaptiveavgpool(x)
        _ = self.tracker_adaptiveavgpool(x)
        x = self.classifier(x)
        _ = self.tracker_classifier(x)
        _ = self.tracker_classifier_softmax(F.softmax(x, 1))
        x = torch.flatten(x, 1)

        return x