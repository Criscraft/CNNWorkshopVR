import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

import Scripts.PredefinedFilterModules as pfm
from Scripts.TrackingModules import ActivationTracker

class SpecialTranslationNet(nn.Module):
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
        blockconfig_list: list = [
            {'n_channels_in' : 1 if i==0 else 16,
            'n_channels_out' : 16, # n_channels_out % shuffle_conv_groups == 0 and n_channels_out % n_classes == 0 
            'conv_groups' : 16 // 4,
            'avgpool' : True if i in [0, 2] else False,
            'filter_mode' : "Uneven",
            'roll_instead_of_3x3': False,
            'randomroll' : -1,
            } for i in range(4)],
        init_mode: str = 'uniform',
        statedict: str = '',
        ):
        super().__init__()

        self.embedded_model = PredefinedFilterNet_(
            n_classes=n_classes,
            start_config=start_config,
            blockconfig_list=blockconfig_list, 
            init_mode=init_mode)

        self.tracker_module_groups_info = {group.meta['group_id'] : {key : group.meta[key] for key in ['precursors', 'label']} for group in pfm.tm.tracker_module_groups}

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
        with torch.no_grad():
            for m in self.modules():
                if hasattr(m, "weight"):
                    if hasattr(m, "weight_limit_min"):
                        m.weight.data = torch.clamp(m.weight.data, m.weight_limit_min, m.weight_limit_max)

    
class PredefinedFilterNet_(nn.Module):

    def __init__(
        self,
        n_classes: int,
        start_config: dict,
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

        tm.instance_tracker_module_group(label="First Convolution")
        # self.conv1 = pfm.PredefinedFilterModule3x3Part(
        #     n_channels_in=start_config['n_channels_in'],
        #     filter_mode=pfm.ParameterizedFilterMode[start_config['filter_mode']],
        #     n_angles=start_config['n_angles'],
        #     handcrafted_filters_require_grad=start_config['handcrafted_filters_require_grad'],
        #     f=start_config['f'],
        #     k=start_config['k'],
        #     stride=start_config['stride'],
        #     activation_layer=nn.ReLU,
        # )

        # Blocks
        blocks = [
            pfm.SpecialTranslationBlock(
                n_channels_in=config['n_channels_in'],
                n_channels_out=config['n_channels_out'],
                conv_groups=config['conv_groups'],
                avgpool=config['avgpool'],
                filter_mode = config['filter_mode'],
                roll_instead_of_3x3=config['roll_instead_of_3x3'],
                randomroll=config['randomroll']
            ) for config in blockconfig_list]
        self.blocks = nn.Sequential(*blocks)

        # AdaptiveAvgPool
        tm.instance_tracker_module_group(label="AvgPool")
        #self.adaptiveavgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.tracker_adaptiveavgpool = tm.instance_tracker_module(label="AvgPool")

        # Classifier
        n_channels_in = blockconfig_list[-1]['n_channels_out']
        self.classifier = nn.Conv2d(n_channels_in, n_classes, kernel_size=1, stride=1, bias=False)
        # For the regularization we have to give the classifier weight_limit_min and weight_limit_max
        self.classifier.weight_limit_min = -n_channels_in
        self.classifier.weight_limit_max = n_channels_in
        tm.instance_tracker_module_group(label="Classifier")
        self.tracker_classifier = tm.instance_tracker_module(label="Classifier", tracked_module=self.classifier, channel_labels="classes")
        self.tracker_classifier_softmax = tm.instance_tracker_module(label="Class Probabilities", channel_labels="classes")
        
        pfm.initialize_weights(self.modules(), init_mode)
        

    def forward(self, x: Tensor) -> Tensor:
        _ = self.tracker_input(x)
        #x = self.conv1(x)
        x = self.blocks(x)

        #x = self.adaptiveavgpool(x)
        x = x[:,:,x.shape[2]//2,x.shape[3]//2]
        x = x.unsqueeze(2).unsqueeze(3)
        _ = self.tracker_adaptiveavgpool(x)
        x = self.classifier(x)
        _ = self.tracker_classifier(x)
        _ = self.tracker_classifier_softmax(F.softmax(x, 1))
        x = torch.flatten(x, 1)

        return x