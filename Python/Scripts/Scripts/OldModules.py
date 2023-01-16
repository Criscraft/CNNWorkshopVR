import torch
import torch.nn as nn
import torch.nn.functional as F

class CompareGroup(nn.Module):
    def __init__(
        self,
        n_channels: int,
        clamp_limit_min : float = 0.,
        clamp_limit_max : float = 1.,
    ) -> None:
        super().__init__()

        self.clamp_limit_min = clamp_limit_min
        self.clamp_limit_max = clamp_limit_max
        self.tracker_input = tm.instance_tracker_module(label="Input")
        self.tracker_out = tm.instance_tracker_module(label="AND, OR")
        self.merge = MergeModule(n_channels, self.tracker_input.module_id, self.tracker_out.module_id)


    def forward(self, x):
        _ = self.tracker_input(x)
        x_skip = x
        x_and = x[:,::2] * x[:,1::2]
        x_or = x[:,::2] + x[:,1::2]
        x_or = DifferentiableClamp.apply(x_or, self.clamp_limit_min, self.clamp_limit_max)
        x_stacked = zip_tensors([x_and, x_or])
        _ = self.tracker_out(x_stacked)
        x_merged = self.merge(x_skip, x_stacked)
        return x_merged

def zip_tensors(tensors):
    shape = tensors[0].shape
    tensors = [x.unsqueeze(2) for x in tensors]
    out = torch.cat(tensors, 2)
    out = out.reshape((shape[0], -1, shape[2], shape[3]))
    return out


class RollGroupFixed(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

        self.tracker_input = tm.instance_tracker_module(label="Input")
        self.tracker_roll = tm.instance_tracker_module(label="Roll")

        
    def forward(self, x):
        _ = self.tracker_input(x)
        # has to match the filter combinations in 3x3 part
        x1 = torch.roll(x[:,0::4], 1, 3) # right
        x1[:,:,:,0] = x1[:,:,:,1].detach()
        x2 = torch.roll(x[:,1::4], 1, 2) # bottom
        x2[:,:,0,:] = x2[:,:,1,:].detach()
        x3 = torch.roll(x[:,2::4], -1, 3) # left
        x3[:,:,:,-1] = x3[:,:,:,-2].detach()
        x4 = torch.roll(x[:,3::4], -1, 2) # top
        x4[:,:,-1,:] = x4[:,:,-2,:].detach()
        x_stacked = zip_tensors([x1, x2, x3, x4])
        _ = self.tracker_roll(x_stacked)
        return x_stacked


class LearnablePermutationModule(WeightRegularizationModule):
    def __init__(self, n_channels):
        super().__init__()

        self.indices = nn.Parameter(torch.arange(n_channels), False)

        weight = torch.eye(n_channels)
        weight = weight.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight, True)
        internal_weight = torch.zeros(n_channels, n_channels, 1, 1)
        self.internal_weight = nn.Parameter(internal_weight, False)
        self.register_param_to_scale(self.internal_weight, -10., 10., 0., 1.)

        self.create_trackers()

    def create_trackers(self):
        self.tracker_out = tm.instance_tracker_module(label="Learned Perm.", draw_edges=True)
        self.tracker_out.register_data("input_mapping", self.indices)

    def forward(self, x):
        if self.training:
            self.weight_old = self.weight.data.clone().detach()
            out = F.conv2d(x, self.weight)
        else:
            out = x[:,self.indices]

        _ = self.tracker_out(out)
        return out

    def compute_indices(self):
        n_channels = self.internal_weight.shape[0]
        index_dict = {} # output_ind -> input_ind
        weight_flattend = self.internal_weight.flatten()
        # weight_flattend = self.internal_weight.flatten(1)
        # urgency_of_input_inds = torch.argsort(torch.max(weight_flattend, 0)[0], descending=True).cpu().numpy()
        # ranking_of_out_channels = torch.argsort(weight_flattend, 0, descending=True).cpu().numpy()
        # for input_ind in urgency_of_input_inds:
        #     inds = ranking_of_out_channels[:,input_ind]
        #     for ind in inds:
        #         if ind not in index_dict:
        #             index_dict[ind] = input_ind
        #             if self.indices.data[ind].item() != input_ind:
        #                 self.internal_weight[ind, input_ind, 0, 0] = self.internal_weight[ind, input_ind, 0, 0] + 5. 
        #             break
        ranking = torch.argsort(weight_flattend, descending=True).cpu().numpy()
        i = 0
        while len(index_dict) != n_channels:
            rank_ind = ranking[i]
            output_ind = rank_ind // n_channels
            input_ind = rank_ind % n_channels
            if output_ind not in index_dict and input_ind not in index_dict.values():
                index_dict[output_ind] = input_ind
                if self.indices.data[output_ind].item() != input_ind:
                    self.internal_weight[output_ind, input_ind, 0, 0] = self.internal_weight[output_ind, input_ind, 0, 0] + 2.
            i += 1
        indices_new = [index_dict[output_ind] for output_ind in range(n_channels)]
        indices_new = torch.tensor(indices_new, dtype=torch.long, device=self.indices.data.device)
        # if torch.any(indices_new != self.indices.data).item():
        #     nn.init.constant_(self.internal_weight, 0.)
        self.indices.data = torch.tensor(indices_new, dtype=torch.long, device=self.indices.data.device)

    def get_sparse_weight_matrix(self):
        weights_permutation = torch.zeros(self.weight.shape[0], self.weight.shape[0], device=self.weight.data.device)
        for i, j in enumerate(self.indices.data):
            weights_permutation[i,j] = 1.
        weights_permutation = weights_permutation.unsqueeze(2).unsqueeze(3)
        return weights_permutation

    def pre_regularize(self):
        # update internal weights
        delta = self.weight - self.weight_old
        del self.weight_old
        self.internal_weight.data = self.internal_weight + torch.clamp(delta, 0., 0.1)
        # replace weights by sparse weights
        self.compute_indices()
        self.weight.data = self.get_sparse_weight_matrix()


class SparseConv2D(WeightRegularizationModule):
    def __init__(
        self,
        n_channels : int,
        conv_groups : int,
        n_selectors : int, 
        selector_radius : int, # the nth channel before or after the current channel will contribute when n<selector_radius
    ) -> None:
        super().__init__()

        self.conv_groups = conv_groups
        group_size = n_channels // conv_groups
        self.n_selectors = n_selectors
        
        # Initialize trainable internal weights
        # Shape of weight_selection: [n_selectors, n_channels_out, 1 (group_size), filter height, filter width]
        weight_selection = torch.ones(self.n_selectors, n_channels, 1, 1, 1) * 0.49 # must not be 0.5 because the torch.abs(x) function in forward has no gradient for x=0
        self.weight_selection = nn.Parameter(weight_selection, True)
        nn.init.uniform_(self.weight_selection, 0., 1.)
        self.register_param_to_cycle(self.weight_selection, 0., 1.)
        # Shape of weight_group: [n_selectors, 1 (batchsize), n_channels, tensor height, tensor width]
        weight_group = torch.zeros(self.n_selectors, 1, n_channels, 1, 1)
        self.weight_group = nn.Parameter(weight_group, True)
        nn.init.uniform_(self.weight_group, -1., 1.)
        self.register_param_to_clamp(self.weight_group, -group_size, group_size, -1., 1.)

        # Initialize helpers
       
        # shape: [1 (n_selectors), 1 (n_channels_out), group_size, 1 (filter height), 1 (filter width)]
        kernel_positions = torch.zeros(1, 1, group_size)
        for i in range(group_size):
                kernel_positions[:,:,i] = i/(group_size-2.)
        kernel_positions = kernel_positions.unsqueeze(3).unsqueeze(4)
        self.kernel_positions = nn.Parameter(kernel_positions, False)
        self.radius = selector_radius / (group_size-1.)

        self.create_trackers(group_size, n_channels)
        self.relu = TrackedLeakyReLU()

    def create_trackers(self, group_size, n_channels):
        self.tracker = tm.instance_tracker_module(label="Sparse Conv", draw_edges=True, ignore_highlight=False)
        self.tracker.register_data("sparse_conv_weight_selection", self.weight_selection)
        self.tracker.register_data("sparse_conv_weight_selection_limit", [0., 1.])
        self.tracker.register_data("sparse_conv_weight_group", self.weight_group)
        self.tracker.register_data("sparse_conv_weight_group_limit", [-group_size, group_size])
        self.tracker.register_data("radius", self.radius)
        self.tracker.register_data("group_size", group_size)
        in_channels_per_group = n_channels // self.conv_groups
        out_channels_per_group = n_channels // self.conv_groups
        input_mapping = []
        for out_channel in range(n_channels):
            group = out_channel // out_channels_per_group
            input_mapping.append(list(range(group*in_channels_per_group, (group+1)*in_channels_per_group)))
        self.tracker.register_data("input_mapping", input_mapping)
        
        # input_mapping = []
        # in_channels_per_group = n_channels // conv_groups
        # out_channels_per_group = n_channels // conv_groups

        # distances = torch.abs(self.weight_selection - self.kernel_positions)
        # # make distances circular
        # distances = torch.minimum(distances, 1. - distances)
        # distances = distances.flatten(2)

        # for selector in range(n_selectors):
        #     input_mapping_selector = []
        #     for out_channel in range(n_channels):
        #         group = out_channel // out_channels_per_group
        #         input_mapping_group = list(range(group*in_channels_per_group, (group+1)*in_channels_per_group))
        #         selected_index = torch.argmin(distances[selector, out_channel].flatten()).cpu().item()
        #         selected_indices = list(range(selected_index-selector_diameter//2, selected_index+selector_diameter//2+1))
        #         input_mapping_selector.append([input_mapping_group[i] for i in selected_indices])
        #     input_mapping.append(input_mapping_selector)
        # self.tracker_out.register_data("input_mapping", input_mapping)

    def forward(self, x):
        distances = torch.abs(self.weight_selection - self.kernel_positions)
        # make distances circular
        distances = torch.minimum(distances, 1. - distances)
        # Create convolution weights
        # When a distance exeeds the radius, the corresponding channel will not contribute.
        w = 1. - distances / self.radius
        w = DifferentiableClamp.apply(w, 0., 1.)
        tensor_list = []
        for selector in range(self.n_selectors):
            y = F.conv2d(x, w[selector], groups=self.conv_groups)
            y = y * self.weight_group[selector]
            tensor_list.append(y)
        x = torch.sum(torch.stack(tensor_list), 0)
        _ = self.tracker(x)
        x = self.relu(x)
        return x