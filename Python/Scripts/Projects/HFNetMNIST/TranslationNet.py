import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import json
import random
import matplotlib.pyplot as plt


import Scripts.PredefinedFilterModules as pfm
from Scripts.TrackingModules import ActivationTracker

class TranslationNet(nn.Module):
    def __init__(self,
        n_classes: int = 10,
        blockconfig_list: list = [
            {'n_channels_in' : 1 if i==0 else 16,
            'n_channels_out' : 16, # n_channels_out % shuffle_conv_groups == 0
            'conv_groups' : 16 // 4,
            'avgpool' : True if i in [3, 6, 9] else False,
            'spatial_mode' : "parameterized_translation", # one of predefined_filters and parameterized_translation
            'spatial_requires_grad' : True,
            'filter_mode' : "TranslationSmooth", # one of Even, Uneven, All, Random, Smooth, EvenPosOnly, UnevenPosOnly, TranslationSmooth, TranslationSharp
            'n_angles' : 2,
            'translation_k' : 5,
            'randomroll' : -1,
            'normalization_mode' : 'layernorm', # one of batchnorm, layernorm
            'permutation' : 'identity', # one of shifted, identity, disabled
            } for i in range(4)],
        init_mode: str = 'identity', # one of uniform, uniform_translation_as_pfm, zero, identity
        conv_expressions = ["white_stripes_s_0"],
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
            self.channel_positions = [] # one element per block the layer of the node spans
            self.blocks = [] # consecutive list of block indices
            self.precursor_ids = [] # list of precursors (ids)
            self.precursors = [] # list of precursors (nodes)
            self.weights = []
            self.channel_widths = []
            self.color = [255, 255, 255]
            
        
    def create_expressions(self, target_conv_expressions, blockconfig_list, blocks):
        # conv_expressions is a list with conv_expression ids.
        pool_booleans = [blockconfig["avgpool"] for blockconfig in blockconfig_list]
        pool_stage_list = [0] # stores for each block the pool stage
        for pool_boolean in pool_booleans[1:]:
            if pool_boolean:
                pool_stage_list.append(pool_stage_list[-1] + 1)
            else:
                pool_stage_list.append(pool_stage_list[-1])
        n_pool_stages = pool_stage_list[-1]
        
        # For each pool stage get a list of expression ids.
        expressions_in_poolstages = [[] for i in range(n_pool_stages)] # Holds for each pool stage the ids of the expressions that should be generated.
        expression_frontier = target_conv_expressions
        # Traverse the graph and fill expressions_in_poolstages. 
        while expression_frontier:
            conv_expression_id = expression_frontier.pop()
            conv_expression = self.conv_expressions[conv_expression_id]
            if conv_expression_id not in expressions_in_poolstages[conv_expression['pooling_stage']]:
                expressions_in_poolstages[conv_expression['pooling_stage']].append(conv_expression_id)
                expression_frontier.extend(conv_expression['input_conv_expression_ids'])
        
        layerings_in_poolstages = [] # Contains a layering for each poolstage. 
        # layerings_in_poolstages is a list of shape [n_poolstages, n_layers, n_elements]
        # A layering is a list of shape [n_layers, n_elements]
        blocks_of_layers = [] # List of shape [n_layers, n_blocks_in_this_layer]. Contains the block id of each layer

        block_counter = 0
        for poolstage in range(n_pool_stages):
            conv_expression_ids = expressions_in_poolstages[poolstage]

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
            for layer in layering:
                widths = [len(self.conv_expressions[conv_expression_id]['weights']) for conv_expression_id in layer]
                max_width = max(widths)
                blocks_of_layers.append(list(range(block_counter, block_counter + max_width)))
                block_counter += max_width

        # Create a node for each convolutional expression
        block_channel_counts = [0 for i in range(len(blocks))]
        nodes = {}
        for poolstage in range(n_pool_stages):
            for layer_id, layer in enumerate(layerings_in_poolstages[poolstage]):
                for conv_expression_id in layer:
                    node = self.Node()
                    node.id = conv_expression_id
                    node.blocks = blocks_of_layers[layer_id]
                    node.precursor_ids = self.conv_expressions[conv_expression_id]['input_conv_expression_ids']
                    node.weights = self.conv_expressions[conv_expression_id]['weights']
                    node.channel_positions = [block_channel_counts[block] for block in node.blocks]
                    node.channel_widths = []
                    channel_width = 0
                    for i, block in enumerate(node.blocks):
                        if len(node.weights) > i:
                            channel_width = max(channel_width, len(node.weights[i][0]))
                        block_channel_counts[block] += channel_width
                        node.channel_widths.append(channel_width)
                    node.color = [random.randint(0, 255) for _ in range(3)]
                    nodes[conv_expression_id] = node

        # Add precursor modules to the nodes
        for node in nodes.values():
            for precursor in node.precursor_ids:
                node.precursors.append(nodes[precursor])

        # For each conv expression check if the precursor ends at the previous block. If not, add linker expressions (they have identity weights) in between.
        for node in nodes.values():
            node_block = node.blocks[0]
            for precursor in node.precursors:    
                precursor_block = precursor.blocks[0]
                distance = node_block - precursor_block 
                if distance > 1:
                    # create identity blocks
                    precursor_block_new = precursor_block
                    for block_new in range(precursor_block + 1, node_block):
                        precursor_new = nodes[precursor_block_new]
                        node_new = self.Node()
                        node_new.blocks = [block_new]
                        node.precursor_ids = [precursor_block_new]
                        node.weights = [[
                            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], 
                            [0, 0, 0, 0],
                            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                            ]]
                        node.channel_positions = block_channel_counts[block_new]
                        node.channel_widths = [precursor_new.channel_widths[-1]]
                        node.color = [255, 255, 255]
                        nodes[f"identity_{block_new}_{node.id}"] = node
                        block_channel_counts[block_new] += node.channel_widths[-1]
                        precursor_block_new = block_new 

        # Debug:
        # Plot the block layout.
        self.draw_conv_expressions(nodes)

        # Fill the network weights with values according to the conv expressions
        for node in nodes.values():
            
            for stage in range(len(node.weights)):
                weights = node.weights[stage]
                block_ind = node.blocks[stage]
                block = blocks[block_ind]
                start_channel = node.channel_positions[stage]
                
                self.write_weights_into_conv(weights[0], block.conv1x1_1, start_channel, node.color)
                self.write_weights_into_conv(weights[2], block.conv1x1_2, start_channel, node.color)
                
                # process spatial blending
                spatialblend_weights = weights[1]
                spatialblend_tracker_out = block.blend.tracker_out
                spatialblend_module_weights = spatialblend_tracker_out.data["data"]["blend_weight"]
                if "colors" not in spatialblend_tracker_out.data:
                    spatialblend_tracker_out.data["colors"] = [[255, 255, 255] for i in range(spatialblend_module_weights.shape[1])]
                colors_blend = spatialblend_tracker_out.data["colors"]
                for out_channel, single_weight in enumerate(spatialblend_weights):
                    out_channel_shifted = out_channel + start_channel
                    spatialblend_module_weights[0, out_channel_shifted, 0, 0] = single_weight
                    colors_blend[out_channel_shifted] = node.color
            
            # The permutation in the preprocessing has to be adjusted. If there are multiple precursors, the input channels have to be stacked.
            channel_counter = 0
            precursors = [nodes[id] for id in node.precursor_ids]
            block = blocks[node.blocks[0]]
            permutation_indices = block.preprocessing.permutation_module.indices
            for precursor in precursors:
                n_channels_of_precursor = precursor.channel_widths[-1]
                indices = [i for i in range(precursor.channel_positions[-1], precursor.channel_positions[-1] + n_channels_of_precursor)]
                indices = torch.LongTensor(indices, device=permutation_indices.device)
                permutation_indices[channel_counter + node.channel_positions[0] : channel_counter + node.channel_positions[0] + n_channels_of_precursor] = indices
                channel_counter += n_channels_of_precursor


    def draw_conv_expressions(self, nodes, figsize=(7, 7)):
        x = []
        y = []
        colors = []
        for node in nodes.values():
            x.extend(node.blocks)
            y.extend(node.channel_positions)
            colors.extend([[node.color[0] / 255., node.color[1] / 255., node.color[2] / 255.] for _ in range(len(node.blocks))])
        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(x, y, s=10, c=colors, marker='s')
        fig.tight_layout()
        fig.savefig("conv_expression_layout.png")
    

    def write_weights_into_conv(self, weights, conv_and_relu_layer, start_channel, color):
        conv_tracker_out = conv_and_relu_layer.conv1x1.tracker_out
        conv_relu_tracker = conv_and_relu_layer.relu.tracker_out
        conv_module_weights = conv_tracker_out.data["data"]["grouped_conv_weight"]
        if "colors" not in conv_tracker_out.data["data"]:
            conv_tracker_out.data["data"]["colors"] = [[255, 255, 255] for i in range(conv_module_weights.shape[0])]
        colors_conv = conv_tracker_out.data["data"]["colors"]
        if "colors" not in conv_relu_tracker.data["data"]:
            conv_relu_tracker.data["data"]["colors"] = [[255, 255, 255] for i in range(conv_module_weights.shape[0])]
        colors_relu = conv_relu_tracker.data["data"]["colors"] 
        for out_channel, group_weights in enumerate(weights):
            for in_channel, single_weight in enumerate(group_weights):
                out_channel_shifted = out_channel + start_channel
                conv_module_weights[out_channel_shifted, in_channel, 0, 0] = single_weight
                colors_conv[out_channel_shifted] = color
                colors_relu[out_channel_shifted] = color


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
                avgpool=config['avgpool'],
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
        
        if conv_expressions:
            # If conv expressions are used, we assume identity intitalization such that we do not have to care about linking unused blocks.
            assert(init_mode=="identity")
        group_size = blockconfig_list[0]['n_channels_in'] // blockconfig_list[0]['conv_groups']
        conv_expression_manager = ConvExpressionsManager("/nfshome/linse/Documents/CNNWorkshopVR/CNNWorkshopVR/Python/Scripts/Projects/HFNetMNIST/conv_expressions.txt", group_size)
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