import os
import shutil
import torch
import torch.nn.functional as F 
import numpy as np
from enum import Enum
import cv2
import copy
import Scripts.utils as utils
from Scripts.FeatureVisualizerRobust import FeatureVisualizer
from Scripts.ImageResource import ImageResource


class FeatureVisualizationMode(Enum):
    Generating = 0
    Loading = 1

# TODO: module_id is an id, not an index. I need a dictionary to link it with the layers. 
# TODO: dictionary from groupids to groups 

class DLNetwork(object):
    
    def __init__(self, model, device, classification, input_size, softmax=False, class_names=[], cache_path=os.path.join("cache", "network")):
        super().__init__()
        
        self.device = device
        self.classification = classification
        self.softmax = softmax #whether to apply softmax on the classification result
        self.class_names = class_names
        self.cache_path = cache_path
        self.model = model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.feature_visualizer = FeatureVisualizer(target_size=input_size)
        self.module_dict = {}
        self.active_image_resource = ImageResource()
        self.active_noise_image = None

        self.output_tracker_module_id = ""
        self.input_tracker_module_id = ""

        if not os.path.exists(self.cache_path): os.makedirs(self.cache_path)

        self.feature_visualization_path = os.path.join(self.cache_path, 'FeatureVisualizations')
        if not os.path.exists(self.feature_visualization_path): os.makedirs(self.feature_visualization_path)
        self.feature_visualization_mode = FeatureVisualizationMode.Loading

        self.export_path = os.path.join(self.cache_path, 'Export')
        if not os.path.exists(self.export_path): os.makedirs(self.export_path)
        
    
    def initial_forward_pass(self, image_resource : ImageResource):
        with torch.no_grad():
            self.active_image_resource = image_resource
            image = self.active_image_resource.data.to(self.device)
            image = image.unsqueeze(0)
            out_dict = self.model.forward_features({'data' : image})
            module_dict = {}
            # assumtion, that the input tracker is the first one and the output tracker is the last one 
            self.input_tracker_module_id = out_dict['layer_infos'][0]['module_id']
            self.output_tracker_module_id = out_dict['layer_infos'][-1]['module_id']
            # record info about each tracker module
            for item in out_dict['layer_infos']:
                if 'activation' in item:
                    if len(item['activation'].shape) == 4:
                        has_data = True
                    size = item['activation'].shape
                else:
                    has_data = False
                    size = [0]
                module_dict[item['module_id']] = { 
                    'tracker_module_type' : item['tracker_module_type'], 
                    'group_id' : item['group_id'], 
                    'label' : item['label'],
                    'precursors' : item['precursors'],
                    'tracked_module' : item['tracked_module'],
                    'channel_labels' : item['channel_labels'],
                    'activation' : item.get('activation'),
                    'has_data' : has_data,
                    'size' : size,
                }
            for module_info in module_dict.values():
                # Add channel labels if necessary
                if module_info['channel_labels'] == "classes":
                    module_info['channel_labels'] = self.class_names
                else:
                    module_info['channel_labels'] = []

            self.module_dict = module_dict


    def forward_pass(self, image_resource : ImageResource):
        with torch.no_grad():
            self.active_image_resource = image_resource
            image = self.active_image_resource.data.to(self.device)
            image = image.unsqueeze(0)
            out_dict = self.model.forward_features({'data' : image})
            # record info about each tracker module
            for item in out_dict['layer_infos']:
                module = self.module_dict[item['module_id']]
                module['activation'] = item.get('activation')


    def get_architecture(self):
        if not self.module_dict:
            raise ValueError("You have to prepare the input first")
        # get group information
        group_dict = copy.deepcopy(self.model.tracker_module_groups_info)
        for group in group_dict.values():
            group['tracker_module_group_type'] = group['tracker_module_group_type'].name
        # get tracker_module information
        out_module_dict = {module_id : {key : module_info[key] for key in ('tracker_module_type', 'group_id', 'label', 'precursors', 'channel_labels', 'has_data', 'size')} for module_id, module_info in copy.deepcopy(self.module_dict).items()}
        
        for module_id, module_info in out_module_dict.items():
            # convert enum to string
            module_info['tracker_module_type'] = module_info['tracker_module_type'].name
            
            # Add information to special cases
            if module_info['tracker_module_type'] == "GROUPCONV":
                weights = self.module_dict[module_id]['tracked_module'].weight.data.cpu().numpy()
                module_info['weights'] = weights.tolist()
                module_info['weights_min'] = float(weights.min())
                module_info['weights_max'] = float(weights.max())
            elif module_info['tracker_module_type'] == "REWIRE":
                module_info['permutation'] = self.module_dict[module_id]['tracked_module'].data.cpu().numpy().tolist()
            elif module_info['tracker_module_type'] == "HFCONV":
                module = self.module_dict[module_id]['tracked_module']
                module_info['kernels'] = module.w.data.cpu().numpy().tolist()
                module_info['padding'] = module.padding
            
        out = {'group_dict' : group_dict, 'module_dict' : out_module_dict}
        return out


    def get_activation(self, module_id):
        # returns cpu tensor
        if self.module_dict:
            return self.module_dict[module_id]['activation'].cpu()
        else:
            raise RuntimeError("need to do the initial forward pass first")
        

    """
    def get_weights(self, module_id):
        module = self.module_dict[module_id]["tracked_module"]
        if module is not None and hasattr(module, "weight"):
            return module.weight
        else:
            return None
    """


    def get_feature_visualization(self, module_id):
        if not self.module_dict:
            raise RuntimeError("need to do the initial forward pass first")
            
        if module_id == self.input_tracker_module_id:
            return np.zeros((self.module_dict[module_id]["size"][1], 3, 1, 1))
        
        created_images = None
        is_loaded = False
        if self.feature_visualization_mode == FeatureVisualizationMode.Loading:
            created_images = self.try_load_feature_visualization(module_id)
            is_loaded = True
        if created_images is None:
            is_loaded = False
            module = self.module_dict[module_id]["tracked_module"]
            n_channels = self.module_dict[module_id]["size"][1]
            created_images, _ = self.feature_visualizer.visualize(self.model, module, self.device, self.active_image_resource.data, n_channels)
        
        out_images = self.feature_visualizer.export_transformation(created_images)

        if not is_loaded: 
            self.save_feature_visualization(module_id, created_images)
        
        return out_images


    def get_classification_result(self):
        if not self.classification:
            return None, None
        out = self.get_activation(self.output_tracker_module_id)
        out = out.flatten(1)
        if self.softmax:
            out = F.softmax(out, 1)
        out = out * 100.
        out = out[0].cpu().numpy()
        indices_tmp = np.argsort(-out)
        indices_tmp = indices_tmp[:10]
        return out[indices_tmp], indices_tmp


    def delete_feat_vis_cache(self):
        shutil.rmtree(self.feature_visualization_path)
        os.makedirs(self.feature_visualization_path)

    
    def try_load_feature_visualization(self, module_id):
        path = os.path.join(self.feature_visualization_path, f"layer_{module_id}.pt")
        if os.path.exists(path):
            created_images = torch.load(path, 'cpu')
        else:
            created_images = None
        return created_images


    def save_feature_visualization(self, module_id, created_images):
        path = os.path.join(self.feature_visualization_path, f"layer_{module_id}.pt")
        torch.save(created_images, path)

    
    def export(self, image_resource : ImageResource):
        images = None
        if image_resource.mode == ImageResource.Mode.FeatureVisualization:
            images = self.get_feature_visualization(image_resource.module_id)
            images = images[:,np.array([2, 1, 0])] # for sorting color channels
            images = images.transpose([0, 2, 3, 1]) # put channel dimension to last
        elif image_resource.mode == ImageResource.Mode.Activation:
            tensor = self.get_activation(image_resource.module_id)
            # give first element of activations because we do not want to have the batch dimension
            tensor = tensor[0]
            tensor_to_uint_transform = utils.TransformToUint()
            images = tensor_to_uint_transform(tensor, True)
        
        path = os.path.join(self.export_path, f"layer_{image_resource.module_id}")
        if not os.path.exists(path): os.makedirs(path)
        for i, image in enumerate(images):
            cv2.imwrite(os.path.join(path, f"{i}.png"), image)


    def get_channel_labels(self, module_id):
        return self.module_dict[module_id]['channel_labels']


    def set_weights(self, module_id, weights):
        self.module_dict[module_id]['tracked_module'].weight.data = weights


