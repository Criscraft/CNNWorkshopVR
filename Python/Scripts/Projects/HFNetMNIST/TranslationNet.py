import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import json
import random
import matplotlib.pyplot as plt
import os


import Scripts.PredefinedFilterModules as pfm
from Scripts.TrackingModules import ActivationTracker

class TranslationNet(nn.Module):
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
            } for i in range(9)],
        init_mode='zero', # one of uniform, uniform_translation_as_pfm, zero, identity
        conv_expressions = ["stripes_st1", "edges_diag_st0"],
        statedict: str = '',
        ):
        super().__init__()

        self.embedded_model = TranslationNet_(
            n_classes=n_classes,
            blockconfig_list=blockconfig_list, 
            init_mode=init_mode,
            conv_expressions=conv_expressions)

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
                self.block.spatial.activation_layer.tracker_out,
                self.block.blend.tracker_out,
                self.block.tracker_input_conv_2,
                self.block.conv1x1_2.conv1x1.tracker_out,
                self.block.conv1x1_2.relu.tracker_out,
            ]
            if hasattr(self.block.preprocessing.pool, "tracker_out"):
                trackers.append(self.block.preprocessing.pool.tracker_out)

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
                    node.n_channels_out_layer = blocks[block_ind].conv1x1_1.conv1x1.tracker_out.data["data"]["grouped_conv_weight"].shape[0]
                    node.color = self.conv_expressions[conv_expression_id]['color']
                    node.input_channel_labels = conv_expression['input_channel_labels']
                    node.output_channel_labels = conv_expression['output_channel_labels']
                    nodes[conv_expression_id] = node
                if layer:
                    layer_id += 1

        # For each conv expression check if the precursor ends at the previous block. If not, add linker expressions (they have identity weights) in between.
        self.add_linker_nodes(nodes, blocks, block_channel_counts)

        # Fill the network weights with values according to the conv expressions
        for node in nodes.values():
            node.write_colors()
            node.write_weights()
            node.write_channel_labels()

        # Plot the block layout.
        self.draw_conv_expressions(nodes)


    def get_conv_expressions_in_poolstages(self, target_conv_expressions, blockconfig_list):
        pool_booleans = [bool(blockconfig["pool_mode"]) for blockconfig in blockconfig_list]
        pool_stage_list = [0] # stores for each block the pool stage
        for pool_boolean in pool_booleans[1:]:
            if pool_boolean:
                pool_stage_list.append(pool_stage_list[-1] + 1)
            else:
                pool_stage_list.append(pool_stage_list[-1])
        n_pool_stages = pool_stage_list[-1] + 1
        
        # For each pool stage get a list of expression ids.
        conv_expressions_in_poolstages = [[] for i in range(n_pool_stages)] # Holds for each pool stage the ids of the expressions that should be generated.
        expression_frontier = target_conv_expressions
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
                        
                        # Create identity node
                        node_new = self.Node()
                        node_new.id = f"identity_{identity_counter}"
                        identity_counter += 1
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
                        node_new.n_channels_out_layer = node_new.block.conv1x1_1.conv1x1.tracker_out.data["data"]["grouped_conv_weight"].shape[0]
                        node_new.color = precursor.color
                        node_new.output_channel_labels = precursor.output_channel_labels
                        node_new.input_channel_labels = precursor.output_channel_labels

                        nodes[node_new.id] = node_new
                        precursor_new = node_new

                    # relink the last node
                    node.precursors[precursor_index] = node_new


    def draw_conv_expressions(self, nodes, figsize=(7, 7)):
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
            ax.annotate(node.id, xy=(x[i],y[i]), xytext=(-30,30),textcoords="offset points",
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
            for p_ in p:
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
        conv_expressions = [],
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
                pool_mode=config['pool_mode'],
                spatial_mode=config['spatial_mode'],
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
        
        group_size = blockconfig_list[0]['n_channels_in'] // blockconfig_list[0]['conv_groups']
        conv_expression_manager = ConvExpressionsManager(os.path.join(os.path.dirname(os.path.realpath(__file__)), "conv_expressions_8_filters.txt"), group_size)
        with torch.no_grad():
            conv_expression_manager.create_expressions(conv_expressions, blockconfig_list, self.blocks)

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