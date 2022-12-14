import torch.nn as nn
from contextlib import contextmanager
from collections import defaultdict


class TrackerModule(nn.Identity):
    def __init__(self, module_id, group_id, label="", precursors=[-1], channel_labels=[], ignore_highlight=True, draw_edges=False):
        """
        data accepts:
        input_mapping, [n_channels_out]
        grouped_conv_weight, [n_channels_out, group_size, 1, 1]
        grouped_conv_weight_limit, 
        PFModule_kernels, [n_kernels, 1, height, width]
        sparse_conv_weight_selection, [n_selectors, n_channels_out, 1 (group_size), filter height, filter width]
        sparse_conv_weight_selection_limit, 
        sparse_conv_weight_group, [n_selectors, batchsize, n_channels, tensor height, tensor width]
        sparse_conv_weight_group_limit,
        blend_weight, [1, n_channels, 1, 1]
        blend_weight_limit, 
        weight_per_channel, [n_channels, 1, 1, 1]
        weight_per_channel_limit
        """
        super().__init__()
        self.module_id = module_id
        # precursors is a list of module ids
        self.precursors = precursors
        self.data = {
            'module_id' : self.module_id,
            'group_id' : group_id,
            'label' : label,
            'precursors' : precursors, 
            #'tracked_module' : tracked_module,
            'tags' : [],
            'data' : {}, # should be filled by subclass.
            'channel_labels' : channel_labels,
            'activation' : None, # filled by ActivationTracker
        }
        if ignore_highlight:
            self.data['tags'].append("ignore_highlight")
        if draw_edges:
            self.data['tags'].append("draw_edges")

    def register_data(self, name, data):
        self.data["data"][name] = data

    def set_data(self, name, data):
        self.data["data"][name].data = data

    def get_data(self):
        return self.data["data"]

    def add_data_ranges(self):
        weight_names = [
            'grouped_conv_weight',
            'sparse_conv_weight_selection',
            'sparse_conv_weight_group', 
            'blend_weight',
            'weight_per_channel',
        ]
        data = self.data["data"]
        for k in list(data.keys()):
            if k in weight_names:
                v = data[k]
                minimum = v.min().cpu().item()
                maximum = v.max().cpu().item()
                data[k + "_range"] = [minimum, maximum]
        

class TrackerModuleGroup(object):
    def __init__(self, group_id, label="", precursors=[-1]):
        # precursors is a list of group ids
        self.group_id = group_id
        self.precursors = precursors
        self.data = {
            'group_id' : self.group_id,
            'precursors' : precursors,
            'label' : label,
        }


class TrackerModuleProvider(object):
    def __init__(self):
        self.group_id = 0
        self.module_id = 0
        self.tracker_module_groups = []

    def instance_tracker_module_group(self, label="", precursors=[-1]):
        self.group_id += 1
        precursors = [p if p>=0 else self.group_id + p for p in precursors]
        new_instance = TrackerModuleGroup(self.group_id, label, precursors)
        self.tracker_module_groups.append(new_instance)
        return new_instance

    def instance_tracker_module(self, group_id=-1, label="", precursors=[-1], channel_labels=[], ignore_highlight=True, draw_edges=False):
        self.module_id += 1
        precursors = [p if p>=0 else self.module_id + p for p in precursors]
        group_id = group_id if group_id>=0 else self.group_id
        new_instance = TrackerModule(self.module_id, group_id, label, precursors, channel_labels, ignore_highlight, draw_edges)
        return new_instance

    def reset_ids(self):
        self.module_id = 0
        self.group_id = 0
        self.tracker_module_groups = []


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

        def store_data(module_, in_data, out_data):
            layer = LayerInfo(name, in_data)
            self._layer_info_dict[module_].append(layer)

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
            #if isinstance(module, TrackerModule): # Does not work for some reason I do not understand.
            if "TrackerModule" in str(module.__class__):
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


    def collect_stats(self, model, batch, module=None, append_module_data=False):
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
                if append_module_data:
                    module_dict = module.data.copy() # Copy important because we do not want the module_dict to be a reference to the tracker module.
                    module_dict['module'] = module
                else:
                    module_dict = {'module_id' : module.data['module_id']}
                module_dict['activation'] = info_item.in_data[0]
                module_dicts.append(module_dict)
        return output, module_dicts