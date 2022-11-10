from importlib.metadata import requires
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import enum

from typing import Type, Any, Callable, Union, List, Tuple, Optional

from torch.nn.modules import module
from Scripts.ActivationTracker import ActivationTracker, TrackerModule, TrackerModuleType, TrackerModuleGroup, TrackerModuleGroupType

# Working Group Convolution and shuffeling of input channels in HFModule. Also added new modes: UnevenPosOnly and EvenPosOnly.

# global variables to create the tracking modules 
TRACKERMODULECOUNTER = 0
TRACKERMODULEGROUPCOUNTER = 0
TRACKERMODULEGROUPS = []

class THFNet(nn.Module):
    def __init__(self,
        n_classes: int = 102,
        start_config: dict = {
            'k' : 3, 
            'filter_mode' : 'All',
            'n_angles' : 4,
            'n_channels_in' : 3,
            'n_channels_out' : 64,
            'f' : 16,
            'handcrafted_filters_require_grad' : False,
            'shuffle_conv_groups' : 1,
        },
        blockconfig_list: List[dict] = [
            {'k' : 3, 
            'filter_mode_1' : 'All',
            'filter_mode_2' : 'All',
            'n_angles' : 4,
            'n_blocks' : 2,
            'n_channels_in' : max(64, 64 * 2**(i-1)),
            'n_channels_out' : 64 * 2**i,
            'avgpool' : True if i>0 else False,
            'f' : 1,
            'handcrafted_filters_require_grad' : False,
            'shuffle_conv_groups' : 1,
            } for i in range(4)],
        avgpool_after_firstlayer: bool = True,
        init_mode: str = 'kaiming_normal',
        statedict: str = '',
        ):
        super().__init__()

        self.embedded_model = HFNet_(
            n_classes=n_classes,
            start_config=start_config,
            blockconfig_list=blockconfig_list, 
            avgpool_after_firstlayer=avgpool_after_firstlayer,
            init_mode=init_mode)

        global TRACKERMODULEGROUPS
        # save info to class, because a second network instance could change TRACKERMODULEGROUPS
        self.tracker_module_groups_info = {group.meta['group_id'] : {key : group.meta[key] for key in ['tracker_module_group_type', 'precursors', 'label']} for group in TRACKERMODULEGROUPS}

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
        logits, layer_infos = track_modules.collect_stats(self.embedded_model, batch['data'], module)
        out = {'logits' : logits, 'layer_infos' : layer_infos}
        return out

            
    def save(self, statedict_name):
        torch.save(self.state_dict(), statedict_name)


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


class HandcraftedFilterModule(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        filter_mode: ParameterizedFilterMode,
        n_angles: int,
        conv: module = nn.Conv2d,
        f: int = 1,
        k: int = 3,
        stride: int = 1,
        activation_layer=nn.Module,
        handcrafted_filters_require_grad: bool = False,
        shuffle_conv_groups: int = 1,
    ) -> None:
        super().__init__()

        # TODO implement shuffeling and store the shuffeling permanently
        self.predev_conv = PredefinedConvnxn(n_channels_in, n_channels_in * f, stride=stride, k=k, filter_mode=filter_mode, n_angles=n_angles, requires_grad=handcrafted_filters_require_grad)
        n_channels_mid = self.predev_conv.n_channels_out
        self.norm1 = LayerNorm()
        self.relu = activation_layer()
        self.conv1x1 = conv(n_channels_mid, n_channels_out, kernel_size=1, stride=1, bias=False, groups=shuffle_conv_groups)

        global TRACKERMODULECOUNTER
        global TRACKERMODULEGROUPCOUNTER
        TRACKERMODULECOUNTER += 1
        self.tracker_input = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.INPUT, TRACKERMODULEGROUPCOUNTER, label="Input")
        TRACKERMODULECOUNTER += 1
        self.tracker_predev_conv = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.HFCONV, TRACKERMODULEGROUPCOUNTER, label="3x3 Conv", precursors=[TRACKERMODULECOUNTER - 1], tracked_module=self.predev_conv)
        TRACKERMODULECOUNTER += 1
        self.tracker_relu = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.RELU, TRACKERMODULEGROUPCOUNTER, label="ReLU", precursors=[TRACKERMODULECOUNTER - 1])
        TRACKERMODULECOUNTER += 1
        self.tracker_norm1 = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.NORM, TRACKERMODULEGROUPCOUNTER, label="LayerNorm", precursors=[TRACKERMODULECOUNTER - 1])
        TRACKERMODULECOUNTER += 1
        self.tracker_conv1x1 = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.GROUPCONV, TRACKERMODULEGROUPCOUNTER, label="1x1 Conv", precursors=[TRACKERMODULECOUNTER - 1], tracked_module=self.conv1x1)


    def forward(self, x: Tensor) -> Tensor:
        _ = self.tracker_input(x)
        x = self.predev_conv(x)
        _ = self.tracker_predev_conv(x)
        x = self.relu(x)
        _ = self.tracker_relu(x)
        x = self.norm1(x)
        _ = self.tracker_norm1(x)
        x = self.conv1x1(x)
        _ = self.tracker_conv1x1(x)
        return x


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


class CopyModule(nn.Module):
    def __init__(
        self,
        n_channels_in : int,
        n_channels_out : int,
    ) -> None:
        super().__init__()

        self.factor = n_channels_out // n_channels_in

    def forward(self, x: Tensor) -> Tensor:
        
        return x.repeat_interleave(2, dim=1)


class LayerNorm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        dims = tuple(range(x.ndim))[1:]
        mean = x.mean(dim=dims, keepdims=True)
        std = x.std(dim=dims, keepdims=True)
        return (x - mean) / (std + 1e-6)


class DoubleHandcraftedFilterModule(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        f: int,
        k: int,
        filter_mode_1: ParameterizedFilterMode,
        filter_mode_2: ParameterizedFilterMode,
        n_angles: int,
        conv: module = nn.Conv2d,
        avgpool: nn.Module = nn.Identity(),
        activation_layer=nn.ReLU,
        handcrafted_filters_require_grad: bool = False,
        shuffle_conv_groups: int = 1,
    ) -> None:
        super().__init__()

        global TRACKERMODULECOUNTER
        global TRACKERMODULEGROUPCOUNTER
        global TRACKERMODULEGROUPS

        # Pooling
        self.avgpool = avgpool
        if isinstance(self.avgpool, nn.Identity):
            self.tracker_avgpool = nn.Identity()
        else:
            TRACKERMODULEGROUPCOUNTER += 1
            TRACKERMODULEGROUPS.append(TrackerModuleGroup(TRACKERMODULEGROUPCOUNTER, TrackerModuleGroupType.POOL, precursors=[TRACKERMODULEGROUPCOUNTER - 1], label="AvgPool"))
            TRACKERMODULECOUNTER += 1
            self.tracker_avgpool = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.POOL, TRACKERMODULEGROUPCOUNTER, label="AvgPool")

        # Permutation
        self.permutation = nn.Parameter(torch.randperm(n_channels_in), False)
        TRACKERMODULEGROUPCOUNTER += 1
        TRACKERMODULEGROUPS.append(TrackerModuleGroup(TRACKERMODULEGROUPCOUNTER, TrackerModuleGroupType.REWIRE, precursors=[TRACKERMODULEGROUPCOUNTER - 1], label="Channel Permutation"))
        TRACKERMODULECOUNTER += 1
        self.tracker_permutation_input = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.INPUT, TRACKERMODULEGROUPCOUNTER, label="Input")
        TRACKERMODULECOUNTER += 1
        self.tracker_permutation_output = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.REWIRE, TRACKERMODULEGROUPCOUNTER, label="Permutation", precursors=[TRACKERMODULECOUNTER - 1], tracked_module=self.permutation)
        summand_index = TRACKERMODULEGROUPCOUNTER
        
        # Copy
        if n_channels_in != n_channels_out:
            self.copymodule = CopyModule(n_channels_in, n_channels_out)
            TRACKERMODULEGROUPCOUNTER += 1
            TRACKERMODULEGROUPS.append(TrackerModuleGroup(TRACKERMODULEGROUPCOUNTER, TrackerModuleGroupType.COPY, precursors=[TRACKERMODULEGROUPCOUNTER - 1], label="Copy channels"))
            summand_index = TRACKERMODULEGROUPCOUNTER
        else:
            self.copymodule = nn.Identity()

        # First HFModule
        TRACKERMODULEGROUPCOUNTER += 1
        TRACKERMODULEGROUPS.append(TrackerModuleGroup(TRACKERMODULEGROUPCOUNTER, TrackerModuleGroupType.HFModule, precursors=[TRACKERMODULEGROUPCOUNTER - 1], label="HFModule 1"))
        
        self.hfmodule_1 = HandcraftedFilterModule(
            n_channels_out,
            n_channels_out,
            f=f,
            k=k,
            filter_mode=filter_mode_1,
            n_angles=n_angles,
            conv=conv,
            stride=1, 
            activation_layer=activation_layer,
            handcrafted_filters_require_grad=handcrafted_filters_require_grad,
            shuffle_conv_groups=shuffle_conv_groups,
            )
        self.relu1 = activation_layer(inplace=False)
        self.norm1 = LayerNorm()

        TRACKERMODULECOUNTER += 1
        self.tracker_relu1 = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.RELU, TRACKERMODULEGROUPCOUNTER, label="ReLU", precursors=[TRACKERMODULECOUNTER - 1])
        TRACKERMODULECOUNTER += 1
        self.tracker_norm1 = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.NORM, TRACKERMODULEGROUPCOUNTER, label="LayerNorm", precursors=[TRACKERMODULECOUNTER - 1])

        # Second HFModule
        TRACKERMODULEGROUPCOUNTER += 1
        TRACKERMODULEGROUPS.append(TrackerModuleGroup(TRACKERMODULEGROUPCOUNTER, TrackerModuleGroupType.HFModule, precursors=[TRACKERMODULEGROUPCOUNTER - 1], label="HFModule 2"))

        self.hfmodule_2 = HandcraftedFilterModule(
            n_channels_out,
            n_channels_out,
            f=f,
            k=k,
            filter_mode=filter_mode_2,
            n_angles=n_angles,
            conv=conv,
            stride=1,
            activation_layer=activation_layer,
            handcrafted_filters_require_grad=handcrafted_filters_require_grad,
            shuffle_conv_groups=shuffle_conv_groups,
            )

        # Skip
        TRACKERMODULEGROUPCOUNTER += 1
        TRACKERMODULEGROUPS.append(TrackerModuleGroup(TRACKERMODULEGROUPCOUNTER, TrackerModuleGroupType.SUM, precursors=[summand_index], label="Skip"))

        # Sum
        TRACKERMODULEGROUPCOUNTER += 1
        TRACKERMODULEGROUPS.append(TrackerModuleGroup(TRACKERMODULEGROUPCOUNTER, TrackerModuleGroupType.SUM, precursors=[TRACKERMODULEGROUPCOUNTER - 2, TRACKERMODULEGROUPCOUNTER - 1], label="Sum"))
        TRACKERMODULECOUNTER += 1
        self.tracker_sum_summand1 = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.INPUT, TRACKERMODULEGROUPCOUNTER, label="Summand 1")
        TRACKERMODULECOUNTER += 1
        self.tracker_sum_summand2 = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.INPUT, TRACKERMODULEGROUPCOUNTER, label="Summand 2")
        TRACKERMODULECOUNTER += 1
        self.tracker_sum = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.SIMPLENODE, TRACKERMODULEGROUPCOUNTER, label="Sum", precursors=[TRACKERMODULECOUNTER - 1, TRACKERMODULECOUNTER - 2])

        # relu and norm
        self.relu2 = activation_layer(inplace=False)
        self.norm2 = LayerNorm()

        TRACKERMODULEGROUPCOUNTER += 1
        TRACKERMODULEGROUPS.append(TrackerModuleGroup(TRACKERMODULEGROUPCOUNTER, TrackerModuleGroupType.SIMPLEGROUP, precursors=[TRACKERMODULEGROUPCOUNTER - 1], label="Activation and Norm"))
        TRACKERMODULECOUNTER += 1
        self.tracker_activation_and_norm_input = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.INPUT, TRACKERMODULEGROUPCOUNTER, label="Input")
        TRACKERMODULECOUNTER += 1
        self.tracker_activation_and_norm_1 = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.RELU, TRACKERMODULEGROUPCOUNTER, label="ReLU", precursors=[TRACKERMODULECOUNTER - 1])
        TRACKERMODULECOUNTER += 1
        self.tracker_activation_and_norm_2 = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.NORM, TRACKERMODULEGROUPCOUNTER, label="LayerNorm", precursors=[TRACKERMODULECOUNTER - 1])
        
        


    def forward(self, x: Tensor) -> Tensor:
        x = self.avgpool(x)
        _ = self.tracker_avgpool(x)

        _ = self.tracker_permutation_input(x)
        x = x[:,self.permutation]
        _ = self.tracker_permutation_output(x)

        x = self.copymodule(x)

        out = self.hfmodule_1(x)
        out = self.relu1(out)
        _ = self.tracker_relu1(out)
        out = self.norm1(out)
        _ = self.tracker_norm1(out)

        out = self.hfmodule_2(out)

        _ = self.tracker_sum_summand1(out)
        _ = self.tracker_sum_summand2(x)
        out += x
        _ = self.tracker_sum(out)

        _ = self.tracker_activation_and_norm_input(out)
        out = self.relu2(out)
        _ = self.tracker_activation_and_norm_1(out)
        out = self.norm2(out)
        _ = self.tracker_activation_and_norm_2(out)

        return out


class HFNet_(nn.Module):

    def __init__(
        self,
        n_classes: int = 102,
        start_config: dict = {
            'k' : 3, 
            'filter_mode' : 'All',
            'n_angles' : 4,
            'n_channels_in' : 3,
            'n_channels_out' : 64,
            'f' : 16,
            'handcrafted_filters_require_grad' : False,
            'shuffle_conv_groups' : 1,
        },
        blockconfig_list: List[Tuple[str]] = [
            {'k' : 3, 
            'filter_mode_1' : 'All',
            'filter_mode_2' : 'All',
            'n_angles' : 4,
            'n_blocks' : 2,
            'n_channels_in' : max(64, 64 * 2**(i-1)),
            'n_channels_out' : 64 * 2**i,
            'avgpool' : True if i>0 else False,
            'f' : 1,
            'handcrafted_filters_require_grad' : False,
            'shuffle_conv_groups' : 1,
            } for i in range(4)],
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

        global TRACKERMODULECOUNTER
        global TRACKERMODULEGROUPCOUNTER
        global TRACKERMODULEGROUPS
        # reset, because a second network instance could change the globals
        TRACKERMODULECOUNTER = 0
        TRACKERMODULEGROUPCOUNTER = 0
        TRACKERMODULEGROUPS = []

        # Input
        TRACKERMODULEGROUPCOUNTER += 1
        TRACKERMODULEGROUPS.append(TrackerModuleGroup(TRACKERMODULEGROUPCOUNTER, TrackerModuleGroupType.INPUT, label="Input"))
        TRACKERMODULECOUNTER += 1
        self.tracker_input_1 = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.INPUT, TRACKERMODULEGROUPCOUNTER, label="Input")

        # First convolution
        TRACKERMODULEGROUPCOUNTER += 1
        TRACKERMODULEGROUPS.append(TrackerModuleGroup(TRACKERMODULEGROUPCOUNTER, TrackerModuleGroupType.HFModule, precursors=[TRACKERMODULEGROUPCOUNTER - 1], label="HFModule"))

        self.conv1 = HandcraftedFilterModule(
            n_channels_in=start_config['n_channels_in'], 
            n_channels_out=start_config['n_channels_out'],
            f=start_config['f'],
            k=start_config['k'],
            filter_mode=ParameterizedFilterMode[start_config['filter_mode']], 
            n_angles=start_config['n_angles'],
            activation_layer=self._activation_layer,
            handcrafted_filters_require_grad=start_config['handcrafted_filters_require_grad'],
            shuffle_conv_groups=start_config['shuffle_conv_groups'],
        )
        self.relu1 = activation_layer(inplace=False)
        self.norm1 = LayerNorm()

        TRACKERMODULECOUNTER += 1
        self.tracker_relu1 = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.RELU, TRACKERMODULEGROUPCOUNTER, label="ReLU", precursors=[TRACKERMODULECOUNTER - 1])
        TRACKERMODULECOUNTER += 1
        self.tracker_norm1 = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.NORM, TRACKERMODULEGROUPCOUNTER, label="LayerNorm", precursors=[TRACKERMODULECOUNTER - 1])
        
        # Pooling
        self.avgpool_after_first_layer = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) if avgpool_after_firstlayer else nn.Identity()

        TRACKERMODULEGROUPCOUNTER += 1
        TRACKERMODULEGROUPS.append(TrackerModuleGroup(TRACKERMODULEGROUPCOUNTER, TrackerModuleGroupType.INPUT, precursors=[TRACKERMODULEGROUPCOUNTER - 1], label="AvgPool"))
        TRACKERMODULECOUNTER += 1
        self.tracker_avgpool_after_first_layer = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.POOL, TRACKERMODULEGROUPCOUNTER, label="AvgPool")

        # Blocks
        self.layers = nn.ModuleList([
            self._make_layer(
                filter_mode_1=ParameterizedFilterMode[config['filter_mode_1']],
                filter_mode_2=ParameterizedFilterMode[config['filter_mode_2']],
                n_angles=config['n_angles'],
                n_blocks=config['n_blocks'],
                n_channels_in=config['n_channels_in'],
                n_channels_out=config['n_channels_out'],
                avgpool=config['avgpool'],
                f=config['f'],
                handcrafted_filters_require_grad=config['handcrafted_filters_require_grad'],
                shuffle_conv_groups=config['shuffle_conv_groups'],
            ) for config in blockconfig_list])

        # AdaptiveAvgPool
        self.adaptiveavgpool = nn.AdaptiveAvgPool2d((1, 1))

        TRACKERMODULEGROUPCOUNTER += 1
        TRACKERMODULEGROUPS.append(TrackerModuleGroup(TRACKERMODULEGROUPCOUNTER, TrackerModuleGroupType.POOL, precursors=[TRACKERMODULEGROUPCOUNTER - 1], label="AvgPool"))
        TRACKERMODULECOUNTER += 1
        self.tracker_adaptiveavgpool = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.POOL, TRACKERMODULEGROUPCOUNTER, label="AvgPool")

        # Classifier
        self.classifier = nn.Conv2d(blockconfig_list[-1]['n_channels_out'], n_classes, kernel_size=1, stride=1, bias=False, groups=n_classes)
        
        TRACKERMODULEGROUPCOUNTER += 1
        TRACKERMODULEGROUPS.append(TrackerModuleGroup(TRACKERMODULEGROUPCOUNTER, TrackerModuleGroupType.GROUPCONV, precursors=[TRACKERMODULEGROUPCOUNTER - 1], label="Classifier"))
        TRACKERMODULECOUNTER += 1
        self.tracker_classifier = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.GROUPCONV, TRACKERMODULEGROUPCOUNTER, label="Classifier", tracked_module=self.classifier, channel_labels="classes")
        TRACKERMODULECOUNTER += 1
        self.tracker_classifier_softmax = TrackerModule(TRACKERMODULECOUNTER, TrackerModuleType.SIMPLENODE, TRACKERMODULEGROUPCOUNTER, label="Class Probabilities", precursors=[TRACKERMODULECOUNTER - 1], channel_labels="classes")

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, PredefinedConv)) and hasattr(m, 'kernel_size') and m.kernel_size==1:
                if init_mode == 'kaiming_normal':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=activation)
                elif init_mode == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity=activation)
                elif init_mode == 'sparse':
                    nn.init.sparse_(m.weight, sparsity=0.1, std=0.01)
                elif init_mode == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(
        self,
        filter_mode_1: ParameterizedFilterMode, 
        filter_mode_2: ParameterizedFilterMode,
        n_angles: int,
        n_blocks: int,
        n_channels_in: int,
        n_channels_out: int,
        f: int,
        k: int = 3,
        avgpool: bool = False,
        handcrafted_filters_require_grad: bool = False,
        shuffle_conv_groups : int = 1,
    ) -> nn.Sequential:

        activation_layer = self._activation_layer
        
        avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) if avgpool else nn.Identity()

        layers = []

        layers.append(DoubleHandcraftedFilterModule(
            n_channels_in=n_channels_in,
            n_channels_out=n_channels_out, 
            f=f,
            k=k,
            filter_mode_1=filter_mode_1,
            filter_mode_2=filter_mode_2,
            n_angles=n_angles,
            avgpool=avgpool,
            activation_layer=activation_layer,
            handcrafted_filters_require_grad=handcrafted_filters_require_grad,
            shuffle_conv_groups=shuffle_conv_groups,
        ))

        for _ in range(1, n_blocks):
            layers.append(DoubleHandcraftedFilterModule(
                n_channels_in=n_channels_out,
                n_channels_out=n_channels_out, 
                f=f,
                k=k,
                filter_mode_1=filter_mode_1,
                filter_mode_2=filter_mode_2,
                n_angles=n_angles,
                activation_layer=activation_layer,
                handcrafted_filters_require_grad=handcrafted_filters_require_grad,
                shuffle_conv_groups=shuffle_conv_groups,
            ))

        return nn.Sequential(*layers)
        

    def forward(self, x: Tensor) -> Tensor:
        _ = self.tracker_input_1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        _ = self.tracker_relu1(x)
        x = self.norm1(x)
        _ = self.tracker_norm1(x)
        x = self.avgpool_after_first_layer(x)
        _ = self.tracker_avgpool_after_first_layer(x)

        for layer in self.layers:
            x = layer(x)

        x = self.adaptiveavgpool(x)
        _ = self.tracker_adaptiveavgpool(x)
        x = self.classifier(x)
        _ = self.tracker_classifier(x)
        _ = self.tracker_classifier_softmax(F.softmax(x, 1))
        x = torch.flatten(x, 1)

        return x