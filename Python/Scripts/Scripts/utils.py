import torch
import numpy as np
import os
import base64
import cv2
import sys
import importlib.util as util 


class TransformToUint(object):

    def __init__(self):
        self.v_max = 0
        self.v_min = 0

    def __call__(self, data, fitting):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if fitting:
            self.v_max = data.max()
            self.v_min = data.min()
            #self.v_max = np.percentile(out, 98)
            #self.v_min = np.percentile(out, 2)

        
        
        data = (data - self.v_min) / (self.v_max - self.v_min + 1e-6)
        # boost dark regions
        stretch_constant = 10
        #data = data * stretch_constant
        #data = np.log(data + 1) / np.log(stretch_constant + 1)
        
        data = (data*255.).clip(0, 255)
        data = data.astype("uint8")
        return data
        

def tensor_to_string(x):
    # tensor has shape (C, H, W) with channels RGB
    if not isinstance(x, np.ndarray):
        x = x.detach().cpu().numpy()
        x = x * 255
        x = x.astype("uint8")
    # if image.shape[0] == 1:
    #     image.expand(-1, 3, -1, -1)
    #if image.shape[0] == 3:
    #    image = image[np.array([2,1,0])] # for sorting color channels to a unity friendly format BGR
    x = x.transpose([1,2,0]) # put channel dimension to last, new shape (H, W, C)
    image_enc = encode_image(x)
    return image_enc


def encode_image(data):
    _, buffer = cv2.imencode('.png', data)
    return base64.b64encode(buffer).decode('utf-8')


def get_module(module_path):
    module_name = os.path.basename(module_path)[:-3]
    
    #if module_path[0]!="/":
    #    module_path = '/'.join([os.getcwd(), module_path])
    
    spec = util.spec_from_file_location(module_name, module_path)
    module = util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def normalize(activations):
    # Normalize such that the values are between 0 and 1
    minimum = activations.min()
    maximum = activations.max()
    activations = (activations - minimum) / (maximum - minimum + 1e-8)
    return activations, minimum, maximum