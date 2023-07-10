import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


import PredefinedFilterModules as pfm
from TrackingModules import ActivationTracker


class ReLUNet(nn.Module):
    def __init__(self, activation : str = "relu"):
        super().__init__()

        self.embedded_model = ReLUNet_(activation=activation)

        self.tracker_module_groups_info = {group.data['group_id'] : {key : group.data[key] for key in ['precursors', 'label']} for group in pfm.tm.tracker_module_groups}

        for m in self.modules():
            if hasattr(m, "add_data_ranges"):
                m.add_data_ranges()
                

    def forward(self, batch):
        if isinstance(batch, dict) and 'data' in batch:
            logits = self.embedded_model(batch['data'])
            out = {'logits' : logits}
            return out
        else:
            return self.embedded_model(batch)


    def forward_features(self, batch, module=None, append_module_data=False):
        track_modules = ActivationTracker()

        assert isinstance(batch, dict) and 'data' in batch
        output, module_dicts = track_modules.collect_stats(self.embedded_model, batch['data'], module, append_module_data)
        out = {'logits' : output, 'module_dicts' : module_dicts}
        return out

            
    def save(self, statedict_name):
        torch.save(self.state_dict(), statedict_name)


    def regularize(self):
        with torch.no_grad():
            for m in self.modules():
                if hasattr(m, "regularize_params"):
                    m.regularize_params()


    def toggle_relus(self, mode : bool):
        for module in self.modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
                if not hasattr(module, "forward_original"):
                    module.forward_original = module.forward
                if mode:
                    module.forward = module.forward_original
                else:
                    module.forward = lambda x : x


    def set_neg_slope_of_leaky_relus(self, slope : float):
        for module in self.modules():
            if isinstance(module, nn.LeakyReLU):
                module.negative_slope = slope


class ReLUNet_(nn.Module):

    def __init__(self, activation : str = "relu") -> None:
        super().__init__()
        
        # reset, because a second network instance could change the globals
        # Reset the id system of the activaion tracker. This is necessary when multiple networks are instanced.
        tm = pfm.tm
        tm.reset_ids()

        # Input
        tm.instance_tracker_module_group(label="Input", precursors=[])
        self.tracker_input = tm.instance_tracker_module(label="Input", precursors=[])

        activation_fn = pfm.TrackedReLU if activation == "relu" else pfm.TrackedLeakyReLU
        self.activation = activation_fn()


    def forward(self, x: Tensor) -> Tensor:
        _ = self.tracker_input(x)
        x = self.activation(x)
        return x


class TranslationNetShort_(nn.Module):

    def __init__(self, activation : str = "relu") -> None:
        super().__init__()
        
        # reset, because a second network instance could change the globals
        # Reset the id system of the activaion tracker. This is necessary when multiple networks are instanced.
        tm = pfm.tm
        tm.reset_ids()

        # Input
        tm.instance_tracker_module_group(label="Input", precursors=[])
        self.tracker_input = tm.instance_tracker_module(label="Input", precursors=[])

        activation_fn = pfm.TrackedReLU if activation == "relu" else pfm.TrackedLeakyReLU
        self.activation = activation_fn()

        self.conv3x3 = pfm.PredefinedConvWithDecorativeCopy(
            n_channels_in=1,
            filter_mode=pfm.ParameterizedFilterMode.TranslationSharp4,
            n_angles=4,
            replicate_padding=True,
        )
        self.conv1x1 = pfm.TrackedConv1x1Regularized(4, 2, 1)



    def forward(self, x: Tensor) -> Tensor:
        _ = self.tracker_input(x)
        x = self.activation(x)
        x = self.conv3x3(x)
        x = self.conv1x1(x)
        return x
    

class TranslationNetLong_(nn.Module):

    def __init__(self, activation : str = "relu") -> None:
        super().__init__()
        
        # reset, because a second network instance could change the globals
        # Reset the id system of the activaion tracker. This is necessary when multiple networks are instanced.
        tm = pfm.tm
        tm.reset_ids()

        # Input
        tm.instance_tracker_module_group(label="Input", precursors=[])
        self.tracker_input = tm.instance_tracker_module(label="Input", precursors=[])

        activation_fn = pfm.TrackedReLU if activation == "relu" else pfm.TrackedLeakyReLU
        self.activation = activation_fn()

        self.conv3x3_1 = pfm.PredefinedConvWithDecorativeCopy(
            n_channels_in=1,
            filter_mode=pfm.ParameterizedFilterMode.Uneven,
            n_angles=2,
            f=4,
            replicate_padding=True,
        )
        self.conv1x1_1 = pfm.TrackedConv1x1Regularized(4, 4, 1)

        self.conv3x3_2 = pfm.PredefinedConvWithDecorativeCopy(
            n_channels_in=2,
            filter_mode=pfm.ParameterizedFilterMode.TranslationSharpLarge4,
            n_angles=4,
            replicate_padding=True,
        )
        self.conv1x1_2 = pfm.TrackedConv1x1Regularized(4, 1, 1)



    def forward(self, x: Tensor) -> Tensor:
        _ = self.tracker_input(x)
        x = self.conv3x3_1(x)
        x = self.conv1x1_1(x)
        x = self.activation(x)
        x = self.conv3x3_2(x)
        x = self.conv1x1_2(x)
        return x