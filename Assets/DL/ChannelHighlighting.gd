extends Node

var module_id_to_highlights = {}

# Called by architecture manager
func initialize():
	var id_to_network_module_resource_dict = get_parent().id_to_network_module_resource_dict
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

# Called via group on_image_selected by ImageSelectButton.
func image_selected(image_resource):
	if image_resource == null:
		return
	if image_resource.mode != ImageResource.MODE.ACTIVATION and image_resource.mode != ImageResource.MODE.FEATURE_VISUALIZATION:
		return
	
	call_deferred("update_highlights", image_resource.module_id, image_resource.channel_id)


func update_highlights(module_id=null, channel_id=null):
	set_highlights_zero()
	
	if module_id == null:
		module_id = module_id_to_highlights.size()
	var id_to_network_module_resource_dict = get_parent().id_to_network_module_resource_dict
	var module_id_to_visits_counter = {}
	for module_id in id_to_network_module_resource_dict:
		module_id_to_visits_counter[module_id] = 0
		
	var frontier_module_ids = []
	frontier_module_ids.append(module_id)
	
	if channel_id != null:
		module_id_to_highlights[module_id][channel_id] = 1.0
	
	var current_module_id
	var current_network_module_resource
	while frontier_module_ids:
		current_module_id = frontier_module_ids.pop_front()
		current_network_module_resource = id_to_network_module_resource_dict[current_module_id]
		for precursor_network_module_resource in current_network_module_resource.precursor_module_resources:
			transfer_highlights(current_network_module_resource, precursor_network_module_resource)
			module_id_to_visits_counter[precursor_network_module_resource.module_id] += 1
			if module_id_to_visits_counter[precursor_network_module_resource.module_id] == precursor_network_module_resource.successor_module_resources.size():
				frontier_module_ids.append(precursor_network_module_resource.module_id)
				
	get_tree().call_group("on_update_highlights", "update_highlights")
		
			
func transfer_highlights(current, precursor):
	var current_highlights = module_id_to_highlights[current.module_id]
	var precursor_highlights = module_id_to_highlights[precursor.module_id]
	var in_channel_ind
	
	if current.weights:
		var group_weights
		var in_channels
		var weight
		
		for out_channel_ind in range(current.weights.size()):
			group_weights = current.weights[out_channel_ind]
			in_channels = current.input_mapping[out_channel_ind]
			for weight_ind in range(group_weights.size()):
				weight = group_weights[weight_ind][0][0]
				in_channel_ind = in_channels[weight_ind]
				precursor_highlights[in_channel_ind] += max(current_highlights[out_channel_ind] * weight, -0.1)
	else:
		for out_channel_ind in range(current.input_mapping.size()):
			in_channel_ind = current.input_mapping[out_channel_ind][0]
			precursor_highlights[in_channel_ind] = current_highlights[in_channel_ind]
