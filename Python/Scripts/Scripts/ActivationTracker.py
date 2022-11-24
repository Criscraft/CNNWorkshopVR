from ast import Add
import torch.nn as nn
from contextlib import contextmanager
from collections import defaultdict
from enum import Enum

# global variables
group_id_ = 0
module_id_ = 0

class TrackerModuleGroup(object):
    def __init__(self, label="", precursors=[-1]):
        # precursors is a list of group ids
        global group_id_
        group_id_ = group_id_ + 1
        self.group_id = group_id_
        precursors = [p if p>=0 else self.group_id + p for p in precursors]
        self.meta = {
            'group_id' : self.group_id,
            'precursors' : precursors,
            'label' : label,
        }


class TrackerModule(nn.Identity):

    def __init__(self, group_id=-1, label="", precursors=[-1], tracked_module=None, info_code="", channel_labels=[]):
        super().__init__()
        # precursors is a list of module ids
        global module_id_
        module_id_ = module_id_ + 1
        self.module_id = module_id_
        precursors = [p if p>=0 else self.module_id + p for p in precursors]
        group_id = group_id if group_id>=0 else group_id_
        self.meta = {
            'module_id' : self.module_id,
            'group_id' : group_id,
            'label' : label,
            'precursors' : precursors, 
            'tracked_module' : tracked_module,
            'info_code' : info_code,
            'channel_labels' : channel_labels,
            'activation' : None,
        }


class LayerInfo():
    def __init__(self, module_name, in_data=None, out_data=None):
        self.module_name = module_name
        #store tensor data, we do not make a copy here and assume that no inplace operations are performed by relus
        self.in_data = in_data
        self.out_data = out_data


class ActivationTracker():
    def __init__(self):
        self._layer_info_dict = None
        
    def register_forward_hook(self, module, name):

        def store_data(module, in_data, out_data):
            layer = LayerInfo(name, in_data)
            self._layer_info_dict[module].append(layer)

        return module.register_forward_hook(store_data)

    """
    def register_forward_hook_finish(self, module, name):

        def store_data(module, in_data, out_data):
            layer = LayerInfo(name, in_data)
            self._layer_info_dict[module].append(layer)
            return torch.ones((1,out_data.shape[1],1,1), device = out_data.device)

        return module.register_forward_hook(store_data)
    """

    @contextmanager
    def record_activations(self, model):
        self._layer_info_dict = defaultdict(list)
        # Important to pass in empty lists instead of initializing
        # them in the function as it needs to be reset each time.
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, TrackerModule):
                handles.append(self.register_forward_hook(module, name))
        yield
        for handle in handles:
            handle.remove()


    @contextmanager
    def record_activation_of_specific_module(self, module):
        self._layer_info_dict = defaultdict(list)
        # Important to pass in empty lists instead of initializing
        # them in the function as it needs to be reset each time.
        handles = []
        handles.append(self.register_forward_hook(module, 'tracked_module'))
        yield
        for handle in handles:
            handle.remove()


    def collect_stats(self, model, batch, module=None):
        if module is not None:
            with self.record_activation_of_specific_module(module):
                output = model(batch)
        else:
            with self.record_activations(model):
                output = model(batch)
        
        #value is a list with one element which is the LayerInfo
        module_dicts = []
        for module, info_list in self._layer_info_dict.items():
            #one info_list can have multiple entries, for example if one relu module is applied several times in a network
            for info_item in info_list:
                module_dict = module.meta
                module_dict['module'] = module
                module_dict['activation'] = info_item.in_data[0]
                module_dicts.append(module_dict)
        return output, module_dicts


def reset_ids():
    global module_id_
    module_id_ = 0
    global group_id_
    group_id_ = 0

"""
module_dicts is a list with module_dicts.
A module_dicts contains:
'module_id' : module_id,
'group_id' : group_id, 
'label' : label,
'precursors' : precursors, 
'tracked_module' : tracked_module,
'info_code' : "",
'channel_labels' : channel_labels,
'input_channels' : -1
'activation' : None,
'module'
"""
