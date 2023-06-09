import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
import os


import PredefinedFilterModules as pfm
from TrackingModules import ActivationTracker


class PreprocessingModule(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int,
        conv_groups: int = 1,
        pool_mode: str = "avgpool",
        norm_module : nn.Module = pfm.TrackedLayerNorm,
        permutation : str = "shifted" # one of shifted, identity, disabled
    ) -> None:
        super().__init__()

        self.n_channels_in = n_channels_in # used to determine if the module copies channels afterwards
        self.n_channels_out = n_channels_out # used to determine if the module copies channels afterwards
        
        tm = pfm.tm
        tm.instance_tracker_module_group(label="Preprocessing")

        # Input tracker
        self.tracker_in = tm.instance_tracker_module(label="Input")

        # Pooling
        self.pool = pfm.TrackedPool(pool_mode) if pool_mode else nn.Identity()

        # Copy channels.
        if n_channels_in != n_channels_out:
            self.copymodule = pfm.CopyModuleNonInterleave(n_channels_in, n_channels_out)
        else:
            self.copymodule = nn.Identity()
            
        # Permutation
        if permutation == "shifted":
            group_size = n_channels_out // conv_groups
            self.permutation_module = pfm.PermutationModule(torch.arange(n_channels_out).roll(group_size // 2))
        elif permutation == "identity":
            self.permutation_module = pfm.PermutationModule(torch.arange(n_channels_out))
        elif permutation == "disabled":
            self.permutation_module = nn.Identity()
        else:
            raise ValueError
        
        # Norm
        if norm_module is not None:
            self.norm_module = norm_module(n_channels_out)
        else:
            self.norm_module = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _ = self.tracker_in(x)
        x = self.pool(x)
        x = self.copymodule(x)
        x = self.permutation_module(x)
        x = self.norm_module(x)
        return x


class TranslationBlock(nn.Module):
    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int, # n_channels_out % shuffle_conv_groups == 0
        conv_groups: int = 1,
        pool_mode: str = "avgpool",
        spatial_mode : str = "predefined_filters", # one of predefined_filters and parameterized_translation
        spatial_requires_grad : bool = True,
        filter_mode: str = "Uneven",
        n_angles : int = 2,
        translation_k : int = 3,
        randomroll: int = -1,
        normalization_mode : str = "layernorm", # one of batchnorm, layernorm
        permutation : str = "shifted", # one of shifted, identity, disabled
        neg_weights_allowed : bool = True,
    ) -> None:
        super().__init__()

        tm = pfm.tm
        
        if normalization_mode == "layernorm":
            norm_module = pfm.TrackedLayerNorm
        elif normalization_mode == "batchnorm":
            norm_module = nn.BatchNorm2d
        elif normalization_mode == "identity":
            norm_module = None
        else:
            raise ValueError

        self.preprocessing = PreprocessingModule(n_channels_in, n_channels_out, conv_groups, pool_mode, norm_module, permutation)

        # 1x1 Conv
        tm.instance_tracker_module_group(label="1x1 Conv")
        self.tracker_input_conv_1 = tm.instance_tracker_module(label="Input")
        self.conv1x1_1 = pfm.Conv1x1AndReLUModule(n_channels_out, n_channels_out, conv_groups, neg_weights_allowed=neg_weights_allowed)

        # Random roll (attack)
        self.randomroll = pfm.RandomRoll(randomroll) if randomroll>0 else nn.Identity()
        
        # Spatial operation
        if spatial_mode == "predefined_filters":
            tm.instance_tracker_module_group(label="3x3 Conv")
            self.tracker_input_spatial = tm.instance_tracker_module(label="Input")
            self.spatial = pfm.PredefinedConvWithDecorativeCopy(
                n_channels_in=n_channels_out,
                filter_mode=pfm.ParameterizedFilterMode[filter_mode],
                n_angles=n_angles,
                filters_require_grad=spatial_requires_grad,
                f=1,
                k=3,
                stride=1,
                replicate_padding=True,
            )
            #self.spatial_norm = norm_module(n_channels_out)
            self.activation = pfm.TrackedLeakyReLU()
        elif spatial_mode == "parameterized_translation":
            tm.instance_tracker_module_group(label="Translation")
            self.tracker_input_spatial = tm.instance_tracker_module(label="Input")
            self.spatial = pfm.ParamTranslationGroup(n_channels_out, translation_k, spatial_requires_grad)
        # Spatial blending (skip)
        self.blend = pfm.BlendModule(n_channels_out, self.tracker_input_spatial.module_id, tm.module_id, monitor_inputs=False)

        # 1x1 Conv
        tm.instance_tracker_module_group(label="1x1 Conv")
        self.tracker_input_conv_2 = tm.instance_tracker_module(label="Input")
        self.conv1x1_2 = pfm.Conv1x1AndReLUModule(n_channels_out, n_channels_out, conv_groups, neg_weights_allowed=neg_weights_allowed)


    def forward(self, x: Tensor) -> Tensor:
        x = self.preprocessing(x)
        _ = self.tracker_input_conv_1(x)
        x = self.conv1x1_1(x)
        x = self.randomroll(x)
        x_skip = x
        _ = self.tracker_input_spatial(x)
        x = self.spatial(x)
        #x = self.spatial_norm(x)
        x = self.activation(x)
        x = self.blend(x_skip, x)
        _ = self.tracker_input_conv_2(x)
        x = self.conv1x1_2(x)
        return x
    

class TranslationNetMNIST(nn.Module):
    def __init__(self,
        n_classes: int = 10,
        blockconfig_list: list = [
            {'n_channels_in' : 1 if i==0 else 16,
            'n_channels_out' : 16, # n_channels_out % shuffle_conv_groups == 0
            'conv_groups' : 16 // 4,
            'pool_mode' : "avgpool" if i in [3, 6] else "",
            'spatial_mode' : "predefined_filters", # one of predefined_filters and parameterized_translation
            'spatial_requires_grad' : False,
            'filter_mode' : "TranslationSharp8", # one of Even, Uneven, All, Random, Smooth, EvenPosOnly, UnevenPosOnly, TranslationSmooth, TranslationSharp4, TranslationSharp8
            'n_angles' : 4,
            'translation_k' : 5,
            'randomroll' : -1,
            'normalization_mode' : 'layernorm', # one of batchnorm, layernorm
            'permutation' : 'identity', # one of shifted, identity, disabled
            'neg_weights_allowed' : True,
            } for i in range(9)],
        init_mode='zero', # one of uniform, uniform_translation_as_pfm, zero, identity
        pool_mode="maxpool",
        conv_expressions = [],
        conv_expressions_path = "conv_expressions_8_filters.txt",
        statedict : str = '',
        freeze_features : bool = False,
        leaky_relu_slope : float = 0.01,
        ):
        super().__init__()

        conv_expressions_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), conv_expressions_path)
        self.embedded_model = TranslationNet_(
            n_classes=n_classes,
            blockconfig_list=blockconfig_list, 
            init_mode=init_mode,
            conv_expressions=conv_expressions,
            conv_expressions_path=conv_expressions_path,
            pool_mode=pool_mode)

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

        self.embedded_model.set_neg_slope_of_leaky_relus(leaky_relu_slope)
                

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

    def freeze_features(self):
        for param in self.embedded_model.parameters():
            param.requires_grad = False
        for param in self.embedded_model.classifier.parameters():
            param.requires_grad = True


class ConvExpressionsManager():
    def __init__(
        self,
        conv_expressions_path : str,
        group_size : int,
    ) -> None:
        with open(conv_expressions_path) as f:
            conv_expressions = json.load(f)
        self.conv_expressions = conv_expressions
        self.group_size = group_size


    class Node():
        def __init__(
            self,
        ) -> None:
            self.id = ""
            self.channel_position = -1 # one element per block the layer of the node spans
            self.block_ind = -1
            self.block = None # block where the node lives
            self.precursors = []
            self.weights = []
            self.n_channels_out = -1
            self.n_channels_out_layer = -1
            self.color = [255, 255, 255]
            self.output_channel_labels = []
            self.input_channel_labels = []


        def write_channel_labels(self):
            trackers_in=[
                self.block.preprocessing.tracker_in,
            ]
            if hasattr(self.block.preprocessing.pool, "tracker_out"):
                trackers_in.append(self.block.preprocessing.pool.tracker_out)

            trackers_in_permuted = [
                self.block.preprocessing.permutation_module.tracker_out,
                self.block.preprocessing.norm_module.tracker_out,
                self.block.tracker_input_conv_1,
            ]

            trackers_out = [
                self.block.conv1x1_2.conv1x1.tracker_out,
                self.block.conv1x1_2.relu.tracker_out,
            ]

            # Add labels of the channel. This code paragraph does not work when a copy module copys channels.
            if self.precursors:
                for precursor in self.precursors:
                    for tracker in trackers_in:
                        if "channel_labels" not in tracker.data['data']:
                            tracker.data['data']["channel_labels"] = ["" for _ in range(precursor.n_channels_out_layer)]
                        labels = tracker.data['data']["channel_labels"]
                        for channel, label in enumerate(precursor.output_channel_labels):
                            out_channel_shifted = channel + precursor.channel_position
                            labels[out_channel_shifted] = label

                for tracker in trackers_in_permuted:
                    if "channel_labels" not in tracker.data['data']:
                        tracker.data['data']["channel_labels"] = ["" for _ in range(self.n_channels_out_layer)]
                    labels = tracker.data['data']["channel_labels"]
                    for channel, label in enumerate(self.input_channel_labels):
                        out_channel_shifted = channel + self.channel_position
                        labels[out_channel_shifted] = label

            for tracker in trackers_out:
                if "channel_labels" not in tracker.data['data']:
                    tracker.data['data']["channel_labels"] = ["" for _ in range(self.n_channels_out_layer)]
                labels = tracker.data['data']["channel_labels"]
                for channel, label in enumerate(self.output_channel_labels):
                    out_channel_shifted = channel + self.channel_position
                    labels[out_channel_shifted] = label
        

        def write_colors(self):
            trackers_in=[
                self.block.preprocessing.tracker_in,
            ]
            if hasattr(self.block.preprocessing.pool, "tracker_out"):
                trackers_in.append(self.block.preprocessing.pool.tracker_out)

            trackers = [
                self.block.preprocessing.permutation_module.tracker_out,
                self.block.preprocessing.norm_module.tracker_out,
                self.block.tracker_input_conv_1,
                self.block.conv1x1_1.conv1x1.tracker_out,
                self.block.conv1x1_1.relu.tracker_out,
                self.block.tracker_input_spatial,
                self.block.spatial.predev_conv.tracker_out,
                #self.block.spatial_norm.tracker_out,
                self.block.blend.tracker_out,
                self.block.tracker_input_conv_2,
                self.block.conv1x1_2.conv1x1.tracker_out,
                self.block.conv1x1_2.relu.tracker_out,
            ]

            for precursor in self.precursors:
                for tracker in trackers_in:
                    if "colors" not in tracker.data['data']:
                        tracker.data['data']["colors"] = [[] for _ in range(precursor.n_channels_out_layer)]
                    colors = tracker.data['data']["colors"]
                    for channel in range(precursor.n_channels_out):
                        out_channel_shifted = channel + precursor.channel_position
                        colors[out_channel_shifted] = precursor.color
                    
            for tracker in trackers:
                if "colors" not in tracker.data['data']:
                    tracker.data['data']["colors"] = [[] for _ in range(self.n_channels_out_layer)]
                colors = tracker.data['data']["colors"]
                for channel in range(self.n_channels_out):
                    out_channel_shifted = channel + self.channel_position
                    colors[out_channel_shifted] = self.color

        def write_weights(self):
            # Permutation
            if self.precursors:
                label_position_dict = {} # stores for each channel label the channel position at the precursor
                for precursor in self.precursors:
                    for i, out_channel_label in enumerate(precursor.output_channel_labels):
                        label_position_dict[out_channel_label] = precursor.channel_position + i
                data = self.block.preprocessing.permutation_module.indices.data
                for channel_ind, label in enumerate(self.input_channel_labels):
                    data[self.channel_position + channel_ind] = label_position_dict[label]

            # conv1
            self.write_weights_into_conv(self.weights[0], self.block.conv1x1_1, self.channel_position)

            # Blend module
            spatialblend_tracker_out = self.block.blend.tracker_out
            spatialblend_module_weights = spatialblend_tracker_out.data["data"]["blend_weight"]
            for out_channel, single_weight in enumerate(self.weights[1]):
                out_channel_shifted = out_channel + self.channel_position
                spatialblend_module_weights[0, out_channel_shifted, 0, 0] = single_weight

            # conv2
            self.write_weights_into_conv(self.weights[2], self.block.conv1x1_2, self.channel_position)


        def write_weights_into_conv(self, weights, conv_and_relu_layer, start_channel):
            conv_tracker_out = conv_and_relu_layer.conv1x1.tracker_out
            conv_module_weights = conv_tracker_out.data["data"]["grouped_conv_weight"]
            for out_channel, group_weights in enumerate(weights):
                out_channel_shifted = out_channel + start_channel
                for in_channel, single_weight in enumerate(group_weights):
                    conv_module_weights[out_channel_shifted, in_channel, 0, 0] = single_weight
        

    def create_expressions(self, target_conv_expressions, blockconfig_list, blocks):
        # target_conv_expressions is a list with conv_expression ids.
        # blockconfig_list is a list with dictionaries. Each dictionary contains information about the network block configuration.
        conv_expressions_in_poolstages, pool_stage_list = self.get_conv_expressions_in_poolstages(target_conv_expressions, blockconfig_list)
        # conv_expressions_in_poolstages is a list with one element per pool stage. Each element is a list of conv expressions.
        # pool_stage_list is a list with the pool stage for each block.
        n_pool_stages = len(conv_expressions_in_poolstages)

        layerings_in_poolstages, blocks_of_layers = self.get_layerings_in_poolstages(conv_expressions_in_poolstages, pool_stage_list)
        # layerings_in_poolstages is a list of shape [n_poolstages, n_layers, n_elements]
        # A layering is a list of shape [n_layers, n_elements]
        # blocks_of_layers is a list of shape [n_layers]. It contains the block id that is on the layer

        # Create a node for each convolutional expression
        block_channel_counts = [0 for i in range(len(blocks))]
        nodes = {}
        layer_id = 0
        for poolstage in range(n_pool_stages):
            for layer in layerings_in_poolstages[poolstage]:
                for conv_expression_id in layer:
                    block_ind = blocks_of_layers[layer_id]
                    conv_expression = self.conv_expressions[conv_expression_id]
                    node = self.Node()
                    node.id = conv_expression_id
                    node.block_ind = block_ind
                    node.block = blocks[block_ind]
                    node.precursors = [nodes[precursor_id] for precursor_id in conv_expression['input_conv_expression_ids']]
                    node.weights = conv_expression['weights']
                    node.n_channels_out = len(node.weights[2])
                    node.channel_position = block_channel_counts[block_ind]
                    block_channel_counts[block_ind] += node.n_channels_out
                    node.n_channels_out_layer = blocks[block_ind].preprocessing.n_channels_out
                    if block_channel_counts[block_ind] > node.n_channels_out_layer:
                        raise ValueError
                    node.color = self.conv_expressions[conv_expression_id]['color']
                    node.input_channel_labels = conv_expression['input_channel_labels']
                    node.output_channel_labels = conv_expression['output_channel_labels']
                    nodes[node.id] = node
                if layer:
                    layer_id += 1

        # create output_nodes
        output_node_ids = []
        for conv_expression in target_conv_expressions:
            precursor = nodes[conv_expression]
            node = self.Node()
            node.id = conv_expression + "_out"
            node.block_ind = len(blocks)
            node.block = None
            node.precursors = [precursor]
            nodes[node.id] = node
            output_node_ids.append(node.id + f"_{node.block_ind}")

        # Change node ids. They should include the block indices.
        for node in nodes.values():
            node.id = node.id + f"_{node.block_ind}"
        nodes = {node.id : node for node in list(nodes.values())}

        # For each conv expression check if the precursor ends at the previous block. If not, add linker expressions (they have identity weights) in between.
        self.add_linker_nodes(nodes, blocks, block_channel_counts)

        # delete superfluous output nodes
        for output_node_id in output_node_ids:
            del nodes[output_node_id]

        # Fill the network weights with values according to the conv expressions
        for node in nodes.values():
            node.write_colors()
            node.write_weights()
            node.write_channel_labels()

        # Plot the block layout.
        self.draw_conv_expressions(nodes)


    def get_pool_stage_list(self, blockconfig_list):
        # pool_stage_list stores for each block the pool stage
        pool_booleans = [bool(blockconfig["pool_mode"]) for blockconfig in blockconfig_list]
        pool_stage_list = [0] 
        for pool_boolean in pool_booleans[1:]:
            if pool_boolean:
                pool_stage_list.append(pool_stage_list[-1] + 1)
            else:
                pool_stage_list.append(pool_stage_list[-1])
        return pool_stage_list


    def get_conv_expressions_in_poolstages(self, target_conv_expressions, blockconfig_list):
        pool_stage_list = self.get_pool_stage_list(blockconfig_list)
        n_pool_stages = pool_stage_list[-1] + 1
        
        # For each pool stage get a list of expression ids.
        conv_expressions_in_poolstages = [[] for i in range(n_pool_stages)] # Holds for each pool stage the ids of the expressions that should be generated.
        expression_frontier = target_conv_expressions.copy()
        # Traverse the graph and fill conv_expressions_in_poolstages. 
        while expression_frontier:
            conv_expression_id = expression_frontier.pop()
            conv_expression = self.conv_expressions[conv_expression_id]
            if conv_expression_id not in conv_expressions_in_poolstages[conv_expression['pooling_stage']]:
                conv_expressions_in_poolstages[conv_expression['pooling_stage']].append(conv_expression_id)
                expression_frontier.extend(conv_expression['input_conv_expression_ids'])

        return conv_expressions_in_poolstages, pool_stage_list
    

    def get_layerings_in_poolstages(self, conv_expressions_in_poolstages, pool_stage_list):
        n_pool_stages = len(conv_expressions_in_poolstages)
        layerings_in_poolstages = [] # Contains a layering for each poolstage. 
        # layerings_in_poolstages is a list of shape [n_poolstages, n_layers, n_elements]
        # A layering is a list of shape [n_layers, n_elements]
        blocks_of_layers = [] # The ith number is the block id that corresponds to layer i. List of shape [n_layers]

        block_counter = 0
        for poolstage in range(n_pool_stages):
            conv_expression_ids = conv_expressions_in_poolstages[poolstage]

            # Now we know which expressions are in which pool stages. Each set of expressions within one pool stage forms a graph. We layer the expressions such that we know their order.
            # Graph layering
            precursors = {conv_expression_id : [id for id in self.conv_expressions[conv_expression_id]['input_conv_expression_ids'] if id in conv_expression_ids] for conv_expression_id in conv_expression_ids}
            # remove ids not present in the poolstage
            layering = self.get_layering(conv_expression_ids, precursors)
            layerings_in_poolstages.append(layering)
            
            # Now we determine the start block for each layer.
            # block_start is the first block in the pooling stage.
            block_start = pool_stage_list.index(poolstage)
            # If we used too many blocks in the old pooling stage, raise an error. The network has to few blocks in that pooling stage.
            if block_counter > block_start:
                raise ValueError(f"The network is to short. Need {block_counter-block_start} more blocks in pooling stage {poolstage}.")
            # the next layers will start at block_start.
            block_counter = block_start
            for layer in layering:
                if not layer:
                    #The layer does not contain any nodes. This means that the subsequent layers will have no nodes, either.
                    break
                # Store the block this layer uses and increase the block counter 
                blocks_of_layers.append(block_counter)
                block_counter += 1

        return layerings_in_poolstages, blocks_of_layers


    def add_linker_nodes(self, nodes, blocks, block_channel_counts):
        # For each conv expression check if the precursor ends at the previous block. If not, add linker expressions (they have identity weights) in between.
        identity_counter = 0
        for node in list(nodes.values()):
            for precursor_index, precursor in enumerate(list(node.precursors)):
                distance = node.block_ind - precursor.block_ind
                
                if distance > 1:
                    precursor_group_size = len(precursor.weights[2][0])
                    precursor_new = precursor
                    
                    for block_new_ind in range(precursor.block_ind + 1, node.block_ind):
                        precursor_new_ind_parts = precursor_new.id.split("_")
                        node_new_id = "_".join(precursor_new_ind_parts[:-1]) + "_" + str(block_new_ind)
                        
                        if node_new_id not in nodes:
                            # Create identity node
                            node_new = self.Node()
                            node_new.id = node_new_id
                            node_new.n_channels_out = precursor.n_channels_out
                            node_new.channel_position = block_channel_counts[block_new_ind]
                            block_channel_counts[block_new_ind] += node_new.n_channels_out
                            node_new.block_ind = block_new_ind
                            node_new.block = blocks[block_new_ind]
                            node_new.precursors = [precursor_new]
                            node_new.weights = [
                                [ [1 if channel % precursor_group_size == group_member else 0 for group_member in range(precursor_group_size) ] for channel in range(precursor.n_channels_out) ],
                                [0 for _ in range(precursor.n_channels_out)],
                                [ [1 if channel % precursor_group_size == group_member else 0 for group_member in range(precursor_group_size) ] for channel in range(precursor.n_channels_out) ],
                                ]
                            node_new.n_channels_out_layer = node_new.block.preprocessing.n_channels_out
                            if block_channel_counts[block_new_ind] > node_new.n_channels_out_layer:
                                raise ValueError
                            node_new.color = precursor.color
                            node_new.output_channel_labels = precursor.output_channel_labels
                            node_new.input_channel_labels = precursor.output_channel_labels
                            nodes[node_new.id] = node_new
                        else:
                            # An identity node for this precursor already exists.
                            node_new = nodes[node_new_id]

                        precursor_new = node_new

                    # relink the last node
                    node.precursors[precursor_index] = node_new


    def draw_conv_expressions(self, nodes, figsize=(9, 9)):
        # TODO: Add annotations, e.g. like in https://stackoverflow.com/questions/7908636/how-to-add-hovering-annotations-to-a-plot
        x = []
        y = []
        x_edges = [[], []] # with shape [2, n_edges]
        y_edges = [[], []] # with shape [2, n_edges]
        colors = []
        for node in nodes.values():
            x.append(node.block_ind)
            y.append(node.channel_position)
            colors.append(node.color)
            for precursor in node.precursors:
                x_edges[0].append(precursor.block_ind)
                y_edges[0].append(precursor.channel_position)
                x_edges[1].append(node.block_ind)
                y_edges[1].append(node.channel_position)
        fig, ax = plt.subplots(figsize=figsize)
        ax.invert_yaxis()
        ax.plot(x_edges, y_edges, linestyle='-', color='black')
        ax.scatter(x, y, s=2000, c=colors, marker='s')
        
        for i, node in enumerate(nodes.values()):
            ax.annotate(node.id, xy=(x[i],y[i]), fontsize=8, xytext=(-8*len(node.id)//4,30),textcoords="offset points",
                bbox=dict(boxstyle="round", fc="w"))


        fig.tight_layout()
        fig.savefig("conv_expression_layout.png")


    def get_layering(self, node_ids, precursors):
        # precursors is a dict with node ids as keys and a list with node ids for the precursors
        q = set(node_ids) # Total set of nodes
        u = set() # Set of nodes that already have been assigned to layers.
        p = set(node_ids) # tracks q - u
        z = set() # Set of nodes that are precursors of the current layer
        current_layer_nodes = []
        selected = False
        layering = []
        
        while not q==u:
            # Update set of nodes which are still unassigned.
            p = p - u
            # Make the algorithm deterministic: In the following for loop we will go through the sorted elements of p.
            p_sorted = list(p)
            p_sorted.sort()
            for p_ in p_sorted:
                if set(precursors[p_]) <= z:
                    # The node p_ has all of its precursors in z, the set of nodes that are precursors of the current layer.
                    # Add the node to the current layer.
                    current_layer_nodes.append(p_)
                    selected = True
                    u.add(p_)
            # If we did not add any node to the current layer, move on to the next layer.
            if not selected:
                layering.append(current_layer_nodes)
                current_layer_nodes = []
                previous_size = len(z)
                z = z | u
                if previous_size == len(z):
                    print("The graph contains a loop and will not be displayed correctly.")
                    break
            selected = False
        # Append the last layer, because it has not been done yet.
        layering.append(current_layer_nodes)
        return layering

class TranslationNet_(nn.Module):

    def __init__(
        self,
        n_classes: int,
        blockconfig_list: list,
        init_mode: str = 'uniform',
        conv_expressions : list = [],
        conv_expressions_path : str = '',
        pool_mode : str = 'avgpool',
    ) -> None:
        super().__init__()
        
        # reset, because a second network instance could change the globals
        # Reset the id system of the activaion tracker. This is necessary when multiple networks are instanced.
        tm = pfm.tm
        tm.reset_ids()

        self.blockconfig_list = blockconfig_list

        # Input
        tm.instance_tracker_module_group(label="Input", precursors=[])
        self.tracker_input = tm.instance_tracker_module(label="Input", precursors=[])

        # Blocks
        blocks = [
            TranslationBlock(
                n_channels_in=config['n_channels_in'],
                n_channels_out=config['n_channels_out'],
                conv_groups=config['conv_groups'],
                pool_mode=config['pool_mode'],
                spatial_mode=config['spatial_mode'],
                spatial_requires_grad = config['spatial_requires_grad'],
                filter_mode=config['filter_mode'],
                n_angles=config['n_angles'],
                translation_k=config['translation_k'],
                randomroll=config['randomroll'],
                normalization_mode=config['normalization_mode'],
                permutation=config['permutation'],
                neg_weights_allowed=config['neg_weights_allowed'],
            ) for config in blockconfig_list]
        self.blocks = nn.Sequential(*blocks)

        # AdaptivePool
        tm.instance_tracker_module_group(label=pool_mode)
        if pool_mode == "avgpool":
            self.adaptivepool = nn.AdaptiveAvgPool2d((1, 1))
        if pool_mode == "maxpool":
            self.adaptivepool = nn.AdaptiveMaxPool2d((1, 1))
        if pool_mode == "lppool":
            self.adaptivepool = pfm.GlobalLPPool(p=4)
        self.tracker_adaptivepool = tm.instance_tracker_module(label=pool_mode)

        # Classifier
        tm.instance_tracker_module_group(label="Classifier")
        n_channels_in = blockconfig_list[-1]['n_channels_out']
        self.classifier = pfm.TrackedConv1x1Regularized(n_channels_in, n_classes, 1)
        self.tracker_classifier_softmax = tm.instance_tracker_module(label="Class Probabilities", channel_labels="classes")
        
        pfm.initialize_weights(self.modules(), init_mode)
        
        group_size = blockconfig_list[0]['n_channels_in'] // blockconfig_list[0]['conv_groups']
        if conv_expressions:
            self.conv_expression_manager = ConvExpressionsManager(conv_expressions_path, group_size)
            with torch.no_grad():
                self.conv_expression_manager.create_expressions(conv_expressions, blockconfig_list, self.blocks)


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


    def set_neg_slope_of_leaky_relus(self, slope : float):
        for module in self.modules():
            if isinstance(module, nn.LeakyReLU):
                module.negative_slope = slope


    def get_pool_stage_list(self, blockconfig_list):
        # Code duplication (see ConvExpressionManager)
        # pool_stage_list stores for each block the pool stage
        pool_booleans = [bool(blockconfig["pool_mode"]) for blockconfig in blockconfig_list]
        pool_stage_list = [0] 
        for pool_boolean in pool_booleans[1:]:
            if pool_boolean:
                pool_stage_list.append(pool_stage_list[-1] + 1)
            else:
                pool_stage_list.append(pool_stage_list[-1])
        return pool_stage_list


    def resize_filter_to_mimic_poolstage_(self, mode : bool):
        if mode:
            poolstage_list = self.get_pool_stage_list(self.blockconfig_list)
            for i, block in enumerate(self.blocks):
                block.spatial.predev_conv.resize_filter_to_mimic_poolstage(poolstage_list[i])
        else:
            for block in self.blocks:
                block.spatial.predev_conv.resize_filter_to_mimic_poolstage(0)                   


    def forward(self, x: Tensor) -> Tensor:
        _ = self.tracker_input(x)
        x = self.blocks(x)

        x = self.adaptivepool(x)
        #x = x[:,:,x.shape[2]//2,x.shape[3]//2]
        #x = x.unsqueeze(2).unsqueeze(3)
        _ = self.tracker_adaptivepool(x)
        x = self.classifier(x)
        _ = self.tracker_classifier_softmax(F.softmax(x, 1))
        x = torch.flatten(x, 1)

        return x