import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import Scripts.PredefinedFilterModules as pfm
from Scripts.TrackingModules import ActivationTracker

class TranslationNet(nn.Module):
    def __init__(self,
        n_classes: int = 10,
        blockconfig_list: list = [
            {'n_channels_in' : 1 if i==0 else 16,
            'n_channels_out' : 16, # n_channels_out % shuffle_conv_groups == 0 and n_channels_out % n_classes == 0 
            'conv_groups' : 16 // 4,
            'avgpool' : True if i in [3, 6, 9] else False,
            'conv_mode' : "sparse", # one of default, sparse
            'sparse_conv_selectors' : 2,
            'sparse_conv_selector_radius' : 1,
            'spatial_mode' : "parameterized_translation", # one of predefined_filters and parameterized_translation
            'spatial_blending' : True,
            'spatial_requires_grad' : True,
            'filter_mode' : "Translation",
            'n_angles' : 2,
            'translation_k' : 5,
            'randomroll' : -1,
            'normalization_mode' : 'layernorm',
            'permutation' : 'static', # one of learnable, static, disabled
            } for i in range(4)],
        init_mode: str = 'uniform',
        statedict: str = '',
        ):
        super().__init__()

        self.embedded_model = TranslationNet_(
            n_classes=n_classes,
            #start_config=start_config,
            blockconfig_list=blockconfig_list, 
            init_mode=init_mode)

        self.tracker_module_groups_info = {group.data['group_id'] : {key : group.data[key] for key in ['precursors', 'label']} for group in pfm.tm.tracker_module_groups}

        if statedict:
            pretrained_dict = torch.load(statedict, map_location=torch.device('cpu'))
            missing = self.load_state_dict(pretrained_dict, strict=True)
            print('Loading weights from statedict. Missing and unexpected keys:')
            print(missing)

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

    
class TranslationNet_(nn.Module):

    def __init__(
        self,
        n_classes: int,
        #start_config: dict,
        blockconfig_list: list,
        init_mode: str = 'uniform',
    ) -> None:
        super().__init__()
        
        # reset, because a second network instance could change the globals
        # Reset the id system of the activaion tracker. This is necessary when multiple networks are instanced.
        tm = pfm.tm
        tm.reset_ids()

        # Input
        tm.instance_tracker_module_group(label="Input", precursors=[])
        self.tracker_input = tm.instance_tracker_module(label="Input", precursors=[])

        # Blocks
        blocks = [
            pfm.TranslationBlock(
                n_channels_in=config['n_channels_in'],
                n_channels_out=config['n_channels_out'],
                conv_groups=config['conv_groups'],
                avgpool=config['avgpool'],
                conv_mode=config['conv_mode'],
                sparse_conv_selectors=config['sparse_conv_selectors'],
                sparse_conv_selector_radius=config['sparse_conv_selector_radius'],
                spatial_mode=config['spatial_mode'],
                spatial_blending = config['spatial_blending'],
                spatial_requires_grad = config['spatial_requires_grad'],
                filter_mode=config['filter_mode'],
                n_angles=config['n_angles'],
                translation_k=config['translation_k'],
                randomroll=config['randomroll'],
                normalization_mode=config['normalization_mode'],
                permutation=config['permutation'],
            ) for config in blockconfig_list]
        self.blocks = nn.Sequential(*blocks)

        # AdaptiveAvgPool
        tm.instance_tracker_module_group(label="AvgPool")
        self.adaptiveavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.tracker_adaptiveavgpool = tm.instance_tracker_module(label="AvgPool")

        # Classifier
        tm.instance_tracker_module_group(label="Classifier")
        n_channels_in = blockconfig_list[-1]['n_channels_out']
        self.classifier = pfm.TrackedConv1x1(n_channels_in, n_classes, 1)
        self.tracker_classifier_softmax = tm.instance_tracker_module(label="Class Probabilities", channel_labels="classes")
        
        pfm.initialize_weights(self.modules(), init_mode)
        

    def forward(self, x: Tensor) -> Tensor:
        _ = self.tracker_input(x)
        x = self.blocks(x)

        x = self.adaptiveavgpool(x)
        #x = x[:,:,x.shape[2]//2,x.shape[3]//2]
        #x = x.unsqueeze(2).unsqueeze(3)
        _ = self.tracker_adaptiveavgpool(x)
        x = self.classifier(x)
        _ = self.tracker_classifier_softmax(F.softmax(x, 1))
        x = torch.flatten(x, 1)

        return x