extends Node

var module_id_to_highlights = {}
var update_required : bool = false
var last_selected_module_id : int = -1
var last_selected_channel_id : int = -1
var debouncing_timer_scene : PackedScene = preload("res://Assets/Stuff/DebouncingTimer.tscn")


func _ready():
	# Add debouncing timer for reacting on changes in network weights
	var timer = debouncing_timer_scene.instance()
	timer.name = "NetworkWeightsDebouncingTimer"
	timer.wait_time = 3
	add_child(timer)
	var _err = timer.connect("timeout", self, "update_highlights", [], CONNECT_DEFERRED)
	add_to_group("on_weight_changed")


# Called by architecture manager
func initialize():
	# initialize module_id_to_highlights
	var id_to_network_module_resource_dict = get_parent().module_id_to_network_module_resource_dict
	var network_module_resource
	for module_id in id_to_network_module_resource_dict:
		network_module_resource = id_to_network_module_resource_dict[module_id]
		module_id_to_highlights[module_id] = []
		for _channel_ind in range(network_module_resource.size[1]):
			module_id_to_highlights[module_id].append(0.0)


func set_highlights_zero():
	var channel_highlights
	for module_id in module_id_to_highlights:
		channel_highlights = module_id_to_highlights[module_id]
		for channel_id in range(channel_highlights.size()):
			channel_highlights[channel_id] = 0.0
			
			
func set_highlights_zero_and_apply():
	set_highlights_zero()
	update_required = false
	last_selected_module_id = -1
	last_selected_channel_id = -1
	get_tree().call_group("on_update_highlights", "update_highlights")
			

# Called via group on_image_selected by ImageSelectButton.
func image_selected(image_resource):
	if image_resource == null:
		return
	if image_resource.mode != ImageResource.MODE.ACTIVATION and image_resource.mode != ImageResource.MODE.FEATURE_VISUALIZATION:
		return
	call_deferred("update_highlights", image_resource.module_id, image_resource.channel_id)


func update_highlights(module_id : int = -1, channel_id : int = -1):
	if module_id >= 0 and channel_id >= 0:
		last_selected_module_id = module_id
		last_selected_channel_id = channel_id
	else:
		module_id = last_selected_module_id
		channel_id = last_selected_channel_id
	
	if module_id < 0 or channel_id < 0:
		set_highlights_zero_and_apply()
		return
		
	set_highlights_zero()
	
	var id_to_network_module_resource_dict = get_parent().module_id_to_network_module_resource_dict
	# For each module we count the number of successors that have been visited.
	var module_id_to_visits_counter = {}
	for module_id in id_to_network_module_resource_dict:
		module_id_to_visits_counter[module_id] = 0
	# We also need the number of successors that need to be visited before we can explore the node further.
	var module_id_to_n_successors = {}
	# To achieve that we have to determine the subgraph that is defined by the input module_id.
	var subgraph_module_ids = []
	# First graph propagation to determine the subgraph.
	var frontier_module_ids = []
	frontier_module_ids.append(module_id)
	subgraph_module_ids.append(module_id)
	var current_module_id
	var current_network_module_resource
	while frontier_module_ids:
		current_module_id = frontier_module_ids.pop_front()
		current_network_module_resource = id_to_network_module_resource_dict[current_module_id]
		for precursor_id in current_network_module_resource.precursors:
			if not precursor_id in subgraph_module_ids:
				# The precursor has not been visited yet.
				subgraph_module_ids.append(precursor_id)
				frontier_module_ids.append(precursor_id)
	# For each module determine the actual number of successors.
	for module_id in id_to_network_module_resource_dict:
		module_id_to_n_successors[module_id] = 0
		for successor_module_resource in id_to_network_module_resource_dict[module_id].successor_module_resources:
			if successor_module_resource.module_id in subgraph_module_ids:
				module_id_to_n_successors[module_id] += 1
				
	# Second graph propagation for the highlight values.
	frontier_module_ids = []
	frontier_module_ids.append(module_id)
	module_id_to_highlights[module_id][channel_id] = 1.0
	while frontier_module_ids:
		current_module_id = frontier_module_ids.pop_front()
		current_network_module_resource = id_to_network_module_resource_dict[current_module_id]
		for precursor_network_module_resource in current_network_module_resource.precursor_module_resources:
			transfer_highlights(current_network_module_resource, precursor_network_module_resource)
			module_id_to_visits_counter[precursor_network_module_resource.module_id] += 1
			if module_id_to_visits_counter[precursor_network_module_resource.module_id] == module_id_to_n_successors[precursor_network_module_resource.module_id]:
				frontier_module_ids.append(precursor_network_module_resource.module_id)
	
	update_required = false
	get_tree().call_group("on_update_highlights", "update_highlights")
		
			
func transfer_highlights(current, precursor):
	var current_highlights = module_id_to_highlights[current.module_id]
	var precursor_highlights = module_id_to_highlights[precursor.module_id]
	
	if "ignore_highlight" in current.tags:
		perform_standard_mapping(current, current_highlights, precursor_highlights)
	elif "grouped_conv_weight" in current.data:
		var weights = current.data["grouped_conv_weight"]
		var input_mapping = current.data["input_mapping"]
		for out_channel_ind in range(weights.size()):
			var group_weights = weights[out_channel_ind]
			var in_channels = input_mapping[out_channel_ind]
			for weight_ind in range(group_weights.size()):
				var weight = group_weights[weight_ind][0][0]
				var in_channel_ind = in_channels[weight_ind]
				precursor_highlights[in_channel_ind] += max(current_highlights[out_channel_ind] * weight, -0.1)
	elif "blend_weight" in current.data:
		# We have a blend module.
		# Determine, if precursor is the first or the second input module.
		var weights = current.data["blend_weight"][0]
		var is_first_module = current.precursors[0] == precursor.module_id
		for out_channel_ind in range(weights.size()):
			var weight = weights[out_channel_ind][0][0]
			if is_first_module:
				precursor_highlights[out_channel_ind] += max(current_highlights[out_channel_ind] * (1.0 - weight), -0.1)
			else:
				precursor_highlights[out_channel_ind] += max(current_highlights[out_channel_ind] * weight, -0.1)
	elif "sparse_conv_weight_selection" in current.data:
		pass
	elif "input_mapping" in current.data:
		perform_standard_mapping(current, current_highlights, precursor_highlights)
		var input_mapping = current.data["input_mapping"]
		for out_channel_ind in range(current.size[1]):
			var in_channel_ind = input_mapping[out_channel_ind]
			precursor_highlights[in_channel_ind] += max(current_highlights[out_channel_ind], -0.1)
	# TODO implement SparseConv2D
				

func perform_standard_mapping(current, current_highlights, precursor_highlights):
	if "input_mapping" in current.data:
		var input_mapping = current.data["input_mapping"]
		for out_channel_ind in range(current.size[1]):
			var in_channel_ind = input_mapping[out_channel_ind]
			precursor_highlights[in_channel_ind] += max(current_highlights[out_channel_ind], -0.1)
	else:
		for out_channel_ind in range(current.size[1]):
			precursor_highlights[out_channel_ind] += max(current_highlights[out_channel_ind], -0.1)
			
# Called via group when a weight is changed.
func weight_changed(_network_module_resource):
	update_required = true
	$NetworkWeightsDebouncingTimer._on_trigger()
