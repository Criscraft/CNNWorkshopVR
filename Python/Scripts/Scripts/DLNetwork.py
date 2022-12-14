import os
import torch
import torch.nn.functional as F 
import numpy as np
import cv2
import copy
import Scripts.utils as utils
from Scripts.FeatureVisualizerRobust import FeatureVisualizer, FeatureVisualizationParams
from Scripts.ImageResource import ImageResource

class DLNetwork(object):
    
    def __init__(self, model, device, classification, input_size, softmax=False, class_names=[], export_path=os.path.join("export", "network")):
        super().__init__()
        
        self.device = device
        self.classification = classification
        self.softmax = softmax #whether to apply softmax on the classification result
        self.class_names = class_names
        self.model = model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.feature_visualizer = FeatureVisualizer()
        self.feature_visualizer.fv_settings.target_size = input_size
        self.module_dict = {}

        self.output_tracker_module_id = ""
        self.input_tracker_module_id = ""

        self.export_path = export_path
        if not os.path.exists(self.export_path): os.makedirs(self.export_path)
        
    
    def initial_forward_pass(self, image_resource : ImageResource):
        with torch.no_grad():
            image = image_resource.data.to(self.device)
            image = image.unsqueeze(0)
            out_dict = self.model.forward_features({'data' : image}, append_module_data=True)
            module_dict = {}
            
            # assumtion, that the input tracker is the first one and the output tracker is the last one 
            self.input_tracker_module_id = out_dict['module_dicts'][0]['module_id']
            self.output_tracker_module_id = out_dict['module_dicts'][-1]['module_id']
            
            # record info about each tracker module
            for item in out_dict['module_dicts']:
                item['size'] = item['activation'].shape
                if item['channel_labels'] == "classes":
                    item['channel_labels'] = self.class_names
                else:
                    item['channel_labels'] = []
                module_dict[item['module_id']] = item

            self.module_dict = module_dict


    def forward_pass(self, image_resource : ImageResource):
        with torch.no_grad():
            image = image_resource.data.to(self.device)
            image = image.unsqueeze(0)
            out_dict = self.model.forward_features({'data' : image}, append_module_data=False)
            # record info about each tracker module
            for sub_module_dict_new in out_dict['module_dicts']:
                sub_module_dict = self.module_dict[sub_module_dict_new['module_id']]
                sub_module_dict['activation'] = sub_module_dict_new['activation'].cpu()


    def get_architecture(self):
        if not self.module_dict:
            raise ValueError("You have to prepare the input first")
        # get group information
        group_dict = copy.deepcopy(self.model.tracker_module_groups_info)
        # get tracker_module information
        out_module_dict = {module_id : {key : copy.deepcopy(module_info[key]) for key in ('group_id', 'label', 'precursors', 'tags', 'data', 'channel_labels', 'size')} for module_id, module_info in self.module_dict.items()}

        # get module information
        for module_info in out_module_dict.values():
            data = module_info["data"]
            if data:
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = v.cpu().numpy().tolist()
                    if isinstance(v, np.ndarray):
                        data[k] = v.tolist()
        out = {'group_dict' : group_dict, 'module_dict' : out_module_dict}
        return out


    def get_activation(self, module_id):
        # returns cpu tensor
        if self.module_dict:
            return self.module_dict[module_id]['activation']
        else:
            raise RuntimeError("need to do the initial forward pass first")


    def get_feature_visualization(self, module_id, image):
        if not self.module_dict:
            raise RuntimeError("need to do the initial forward pass first")
            
        if module_id == self.input_tracker_module_id:
            return np.zeros((self.module_dict[module_id]["size"][1], 3, 1, 1))
        
        module = self.module_dict[module_id]["module"]
        n_channels = self.module_dict[module_id]["size"][1]
        created_images, _ = self.feature_visualizer.visualize(self.model, module, self.device, image, n_channels)
        out_images = self.feature_visualizer.export_transformation(created_images)

        return out_images


    def set_feature_visualization_params(self, param_dict):
        fv_settings = FeatureVisualizationParams(**param_dict)
        fv_settings.mode = FeatureVisualizationParams.Mode(param_dict["mode"])
        self.feature_visualizer.fv_settings = fv_settings


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


    def set_data(self, module_id, name, data):
        module_info = self.module_dict[module_id]
        # Inplace operation injects the change of weights into the network.
        module_info["data"][name][:] = data