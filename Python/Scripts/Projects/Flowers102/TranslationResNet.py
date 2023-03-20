import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import Scripts.PredefinedFilterModules as pfm
from Scripts.TrackingModules import ActivationTracker

"""
This implementation of PFMCNNs has small differences to the original implementation: 
It uses LeakyReLU and it uses a differen filter order (which is irrelevant to the results)
The 1x1 conv has a bias in this implementation.
There is an additional 1x1 layer in each basic block such that the nxn conv layers are sandwiched by 1x1 layers.

The implementation of ResNet with dense convolutions is identical to the default one of Pytorch. However, some layers have a bias in this implementation, which is not the case in the Pytorch implementation.
"""


class TranslationResNet(nn.Module):
    def __init__(self,
        n_classes: int = 102,
        first_block_config={
            'spatial_mode' : "predefined_filters", # one of predefined_filters and dense_convolution
            'n_channels_in' : 3,
            'n_channels_out' : 64,
            'k' : 3,
            'stride' : 2,
            'padding': 1,
            'filter_mode' : "EvenAndUneven", # for predefined_filters only
            'n_angles' : 4, # for predefined_filters only
            'filters_require_grad' : False, # for predefined_filters only
            'f' : 16, # for predefined_filters only
            'parameterized_translation' : False,
            'translation_k' : 5, # for parameterized_translation only
            'translation_gradient_radius' : 0, # for parameterized_translation only
        },
        blockconfig_list: list = [
            {'n_channels_in' : 64,
            'n_channels_out' : 64,
            'conv_groups' : 1,
            'pool_mode' : "",
            'k' : 3,
            'stride' : 2 if i in [2, 4, 6] else 1,
            'spatial_mode' : "predefined_filters", # one of predefined_filters and dense_convolution
            'filters_require_grad' : False, # for predefined_filters only
            'filter_mode' : "EvenAndUneven", # for predefined_filters only; one of Even, Uneven, EvenAndUneven, Random, Smooth, EvenPosOnly, UnevenPosOnly, TranslationSmooth, TranslationSharp4, TranslationSharp8
            'n_angles' : 4, # for predefined_filters only
            'parameterized_translation' : False,
            'translation_k' : 5, # for parameterized_translation only
            'translation_gradient_radius' : 0, # for parameterized_translation only
            'randomroll' : 0,
            } for i in range(8)],
        init_mode='kaiming', # one of uniform, zero, identity, kaiming
        first_pool_mode="maxpool", # one of maxpool, avgpool, lppool
        global_pool_mode="avgpool", # one of maxpool, avgpool, lppool
        permutation_mode='disabled', # one of shifted, identity, disabled
        statedict : str = '',
        freeze_features : bool = False,
        ):
        super().__init__()

        self.embedded_model = TranslationNet_(
            n_classes=n_classes,
            first_block_config=first_block_config,
            blockconfig_list=blockconfig_list, 
            init_mode=init_mode,
            first_pool_mode=first_pool_mode,
            global_pool_mode=global_pool_mode,
            permutation_mode=permutation_mode,
            )

        self.tracker_module_groups_info = {group.data['group_id'] : {key : group.data[key] for key in ['precursors', 'label']} for group in pfm.tm.tracker_module_groups}

        if statedict:
            pretrained_dict = torch.load(statedict, map_location=torch.device('cpu'))
            missing = self.load_state_dict(pretrained_dict, strict=True)
            print('Loading weights from statedict. Missing and unexpected keys:')
            print(missing)

        for m in self.modules():
            if hasattr(m, "add_data_ranges"):
                m.add_data_ranges()

        if freeze_features:
            self.freeze_features()
                

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


    def regularize(self): # method name must not be regularize_params because of infinit loop
        with torch.no_grad():
            for m in self.modules():
                if hasattr(m, "regularize_params"):
                    m.regularize_params()

    def update_internals(self): # method name must not be update_internal_weights because of infinit loop
        with torch.no_grad():
            for m in self.modules():
                if hasattr(m, "update_internal_weights"):
                    m.update_internal_weights()

    def freeze_features(self):
        for param in self.embedded_model.parameters():
            param.requires_grad = False
        for param in self.embedded_model.classifier.parameters():
            param.requires_grad = True


class BasicModule(nn.Module):
    def __init__(
        self,
        n_channels_in,
        n_channels_out,
        conv_groups=1,
        spatial_mode="predefined_filters", 
        parameterized_translation=False,
        filters_require_grad=False,
        filter_mode="",
        k=3,
        f=1,
        stride=1,
        padding=1,
        n_angles=4,
        translation_k=5,
        translation_gradient_radius=0,
        randomroll=0
    ) -> None:
        super().__init__()

        tm = pfm.tm

        if randomroll > 0:
            tm.instance_tracker_module_group(label="RandomRoll")
            self.random_roll = pfm.RandomRoll(randomroll)
        else:
            self.random_roll = nn.Identity()

        if parameterized_translation:
            tm.instance_tracker_module_group(label="Param. transl.")
            self.param_translation = pfm.ParamTranslationGroup(n_channels_in, translation_k, translation_gradient_radius, True)
        else:
            self.param_translation = nn.Identity()

        module_list = []

        if spatial_mode == "predefined_filters":
            tm.instance_tracker_module_group(label="Conv3x3")
            module_list.append(tm.instance_tracker_module(label="Input"))
            module_list.append(pfm.PredefinedConvWithDecorativeCopy(
                n_channels_in=n_channels_in,
                filter_mode=pfm.ParameterizedFilterMode[filter_mode],
                n_angles=n_angles,
                filters_require_grad=filters_require_grad,
                f=f,
                k=k,
                stride=stride))
            module_list.append(pfm.TrackedBatchNorm(n_channels_in * f))
            module_list.append(pfm.TrackedLeakyReLU())
            tm.instance_tracker_module_group(label="Conv1x1")
            module_list.append(tm.instance_tracker_module(label="Input"))
            module_list.append(pfm.TrackedConv(n_channels_in * f, n_channels_out, conv_groups, k=1, padding=0, bias=True))
            
        elif spatial_mode == "dense_convolution":
            tm.instance_tracker_module_group(label="Conv3x3")
            module_list.append(tm.instance_tracker_module(label="Input"))
            module_list.append(pfm.TrackedConv(n_channels_in, n_channels_out, conv_groups=conv_groups, k=k, stride=stride, padding=padding))
        
        else:
            raise ValueError
        
        module_list.append(pfm.TrackedBatchNorm(n_channels_out))
        self.conv = nn.Sequential(*module_list)
        

    def forward(self, x):
        x = self.random_roll(x)
        x = self.param_translation(x)
        x = self.conv(x)
        return x


class BasicBlock(nn.Module):
    def __init__(
        self,
        n_channels_in,
        n_channels_out,
        conv_groups=1,
        pool_mode="avgpool",
        spatial_mode="predefined_filters", # one of predefined_filters and dense_convolution
        parameterized_translation=False,
        filters_require_grad=False, # does not apply for dense_convolution
        filter_mode="Uneven", # for predefined_filters only; one of Even, Uneven, All, Random, Smooth, EvenPosOnly, UnevenPosOnly, TranslationSmooth, TranslationSharp4, TranslationSharp8
        k=3,
        stride=1,
        n_angles=4, # for predefined_filters only
        translation_k=5, # for parameterized_translation only
        translation_gradient_radius=0,
        randomroll=0,
        permutation_mode='disabled', # one of shifted, identity, disabled
    ) -> None:
        super().__init__()

        tm = pfm.tm

        if permutation_mode != "disabled":
            tm.instance_tracker_module_group(label="Preprocessing")
        if permutation_mode == "shifted":
            group_size = n_channels_in // conv_groups
            self.permutation = pfm.PermutationModule(torch.arange(n_channels_in).roll(group_size // 2))
        elif permutation_mode == "identity":
            self.permutation = pfm.PermutationModule(torch.arange(n_channels_in))
        elif permutation_mode == "disabled":
            self.permutation = nn.Identity()
        else:
            raise ValueError

        skip_group_id = tm.group_id
        skip_module_id = tm.module_id

        convblock_list = []
        if spatial_mode == "predefined_filters":
            tm.instance_tracker_module_group(label="Conv1x1")
            convblock_list.append(tm.instance_tracker_module(label="Input"))
            convblock_list.append(pfm.TrackedConv(n_channels_in, n_channels_out, conv_groups, k=1, padding=0, bias=True))
            convblock_list.append(pfm.TrackedBatchNorm(n_channels_out))
            convblock_list.append(pfm.TrackedLeakyReLU())

        convblock_list.append(BasicModule(
            n_channels_in=n_channels_out if spatial_mode == "predefined_filters" else n_channels_in,
            n_channels_out=n_channels_out,
            conv_groups=conv_groups,
            spatial_mode=spatial_mode, 
            parameterized_translation=parameterized_translation,
            filters_require_grad=filters_require_grad,
            filter_mode=filter_mode,
            k=k,
            stride=stride,
            padding=k//2,
            n_angles=n_angles,
            translation_k=translation_k,
            translation_gradient_radius=translation_gradient_radius,
            randomroll=randomroll,
        ))
        convblock_list.append(pfm.TrackedLeakyReLU())
        if pool_mode:
            tm.instance_tracker_module_group(label="Pooling")
            convblock_list.append(pfm.TrackedPool(pool_mode, k=3))
        convblock_list.append(BasicModule(
            n_channels_in=n_channels_out,
            n_channels_out=n_channels_out,
            conv_groups=conv_groups,
            spatial_mode=spatial_mode, 
            parameterized_translation=parameterized_translation,
            filters_require_grad=filters_require_grad,
            filter_mode=filter_mode,
            k=k,
            stride=1,
            padding=k//2,
            n_angles=n_angles,
            translation_k=translation_k,
            translation_gradient_radius=translation_gradient_radius,
            randomroll=randomroll,
        ))
        self.convblocks = nn.Sequential(*convblock_list)

        x_group_id = tm.group_id
        x_module_id = tm.module_id

        if pool_mode or n_channels_in != n_channels_out or stride > 1:
            tm.instance_tracker_module_group(label="Skip", precursors=[skip_group_id])
            skip_group_id = tm.group_id
            skip_module_list = []

            if pool_mode:
                skip_module_list.append(pfm.TrackedPool(pool_mode, k=3))
                skip_module_list[-1].tracker_out.precursors = [skip_module_id]
                skip_module_id = tm.module_id

            if stride > 1:
                skip_module_list.append(pfm.TrackedSmoothConv(k=3, n_channels=n_channels_in))
                skip_module_list[-1].tracker_out.precursors = [skip_module_id]
                skip_module_id = tm.module_id
            
            if n_channels_in != n_channels_out or stride > 1:
                #skip_module_list.append(pfm.ChannelPadding(n_channels_in, n_channels_out))
                skip_module_list.append(pfm.TrackedConv(n_channels_in, n_channels_out, conv_groups=conv_groups, k=1, stride=stride, padding=0, bias=True))
                skip_module_list[-1].tracker_out.precursors = [skip_module_id]
                skip_module_list.append(pfm.TrackedBatchNorm(n_channels_out))
                skip_module_id = tm.module_id
                
            self.skip = nn.Sequential(*skip_module_list)

            skip_group_id = tm.group_id
            skip_module_id = tm.module_id
        
        else:
            self.skip = nn.Identity()

        tm.instance_tracker_module_group(label="Sum", precursors=[skip_group_id, x_group_id])
        self.input_tracker_1 = tm.instance_tracker_module(label="X", precursors=[x_module_id])
        self.input_tracker_2 = tm.instance_tracker_module(label="Skip", precursors=[skip_module_id])
        self.tracker_sum = tm.instance_tracker_module(label="Sum", precursors=[self.input_tracker_1.module_id, self.input_tracker_2.module_id])
        self.activation = pfm.TrackedLeakyReLU()
        

    def forward(self, x: Tensor) -> Tensor:
        x = self.permutation(x)
        
        x1 = self.convblocks(x)
        x_skip = self.skip(x)
        
        _ = self.input_tracker_1(x1)
        _ = self.input_tracker_2(x_skip)

        out = x_skip + x1

        _ = self.tracker_sum(out)
        out = self.activation(out)
        return out


class TranslationNet_(nn.Module):

    def __init__(
        self,
        n_classes: int,
        first_block_config: dict,
        blockconfig_list: list,
        init_mode: str = 'uniform',
        first_pool_mode : str = "maxpool", # one of maxpool, avgpool, lppool
        global_pool_mode : str = "avgpool", # one of maxpool, avgpool, lppool
        permutation_mode : str = 'disabled', # one of shifted, identity, disabled
    ) -> None:
        super().__init__()
        
        # reset, because a second network instance could change the globals
        # Reset the id system of the activaion tracker. This is necessary when multiple networks are instanced.
        tm = pfm.tm
        tm.reset_ids()

        # Input
        tm.instance_tracker_module_group(label="Input", precursors=[])
        self.tracker_input = tm.instance_tracker_module(label="Input", precursors=[])

        # First block
        first_block_list = []
        first_block_list.append(BasicModule(**first_block_config))
        
        first_block_list.append(pfm.TrackedLeakyReLU())
        self.first_block = nn.Sequential(*first_block_list)
        
        
        tm.instance_tracker_module_group(label="Pooling")
        self.first_pool = pfm.TrackedPool(pool_mode=first_pool_mode, k=3)

        # Blocks
        blocks = [
            BasicBlock(permutation_mode=permutation_mode, **config) for config in blockconfig_list]
        self.blocks = nn.Sequential(*blocks)

        # AdaptivePool
        tm.instance_tracker_module_group(label=global_pool_mode)
        if global_pool_mode == "avgpool":
            self.adaptivepool = nn.AdaptiveAvgPool2d((1, 1))
        if global_pool_mode == "maxpool":
            self.adaptivepool = nn.AdaptiveMaxPool2d((1, 1))
        if global_pool_mode == "lppool":
            self.adaptivepool = pfm.GlobalLPPool(p=4)
        self.tracker_adaptivepool = tm.instance_tracker_module(label=global_pool_mode)

        # Classifier
        tm.instance_tracker_module_group(label="Classifier")
        n_channels_in = blockconfig_list[-1]['n_channels_out']
        self.classifier = pfm.TrackedLinear(n_channels_in, n_classes)
        
        self.tracker_classifier_softmax = tm.instance_tracker_module(label="Class Probabilities", channel_labels="classes")
        
        pfm.initialize_weights(self.modules(), init_mode)


    def set_tracked_pool_mode_(self, pool_mode):
        for module in self.modules():
            if hasattr(module, "set_tracked_pool_mode"):
                module.set_tracked_pool_mode(pool_mode)


    def toggle_relus(self, mode : bool):
        for module in self.modules():
            if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
                if not hasattr(module, "forward_original"):
                    module.forward_original = module.forward
                if mode:
                    module.forward = module.forward_original
                else:
                    module.forward = lambda x : x

    def resize_filter_to_mimic_poolstage_(self, mode : bool):
        pass                  


    def forward(self, x: Tensor) -> Tensor:
        _ = self.tracker_input(x)
        x = self.first_block(x)
        x = self.first_pool(x)
        x = self.blocks(x)

        x = self.adaptivepool(x)
        _ = self.tracker_adaptivepool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        _ = self.tracker_classifier_softmax(F.softmax(x, 1))

        return x