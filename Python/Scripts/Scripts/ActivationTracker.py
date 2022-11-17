from ast import Add
import torch.nn as nn
from contextlib import contextmanager
from collections import defaultdict
from enum import Enum

class TrackerModuleGroupType(Enum):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"
    GROUPCONV = "GROUPCONV"
    HFModule = "HFModule"
    REWIRE = "REWIRE"
    COPY = "COPY" # induces a split in the computational graph, where channels are copied
    ADD = "ADD"
    POOL = "POOL"
    SUM = "SUM"
    SIMPLEGROUP = "SIMPLEGROUP"

class TrackerModuleType(Enum):
    INPUT = "INPUT" # allows muting
    OUTPUT = "OUTPUT"
    GROUPCONV = "GROUPCONV" # connections inside group only, editable weights
    SIMPLENODE = "SIMPLENODE" # show activations
    REWIRE = "REWIRE" # input - output rewiring
    COPY = "COPY"
    ADD = "ADD" # merge two branches
    POOL = "POOL"
    RELU = "RELU"
    CONV = "CONV"
    HFCONV = "HFCONV" # shows kernel and output, when f>1 one input is linked to several kernels
    NORM = "NORM"


class TrackerModuleGroup(object):
    def __init__(self, id, tracker_module_group_type, label="", precursors=[]):
        self.meta = {
            'group_id' : id,
            'tracker_module_group_type' : tracker_module_group_type,
            'precursors' : precursors,
            'label' : label,
        }


class TrackerModule(nn.Identity):

    def __init__(self, id, tracker_module_type, group_id, label="", precursors=[], tracked_module=None, ignore_activation=False, channel_labels=[], input_channels=-1):
        super().__init__()
        self.meta = {
            'module_id' : id,
            'tracker_module_type' : tracker_module_type,
            'group_id' : group_id, 
            'label' : label,
            'precursors' : precursors, 
            'tracked_module' : tracked_module,
            'ignore_activation' : ignore_activation,
            'channel_labels' : channel_labels,
            'input_channels' : input_channels,
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
                #item_dict['module_id'] = module.module_name
                if not module_dict['ignore_activation']:
                    module_dict['activation'] = info_item.in_data[0]
                module_dicts.append(module_dict)
        return output, module_dicts


"""
module_dicts is a list with module_dicts.
A module_dicts contains:
'module_id' : module_id,
'tracker_module_type' : tracker_module_type,
'group_id' : group_id, 
'label' : label,
'precursors' : precursors, 
'tracked_module' : tracked_module,
'ignore_activation' : ignore_activation,
'channel_labels' : channel_labels,
'input_channels' : -1
'activation' : None,
"""
