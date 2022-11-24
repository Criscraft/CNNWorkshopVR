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
	
	if current.permutation:
		for i in range(current.permutation.size()):
			var j = current.permutation[i]
			precursor_highlights[j] = current_highlights[i]
	elif current.weights:
		var in_channels_per_group = current.weights[0].size()
		var out_channels = current.size[1]
		var in_channels = precursor.size[1]
		var n_groups = in_channels / in_channels_per_group
		var out_channels_per_group = out_channels / n_groups
		for out_channel_ind in range(current.weights.size()):
			var group_weights = current.weights[out_channel_ind]
			var group_ind = int(out_channel_ind / out_channels_per_group)
			for groupmember_ind in range(in_channels_per_group):
				var in_channel_ind = group_ind * in_channels_per_group + groupmember_ind
				var weight = group_weights[groupmember_ind][0][0]
				# if we have a negative contribution, clip it.
				precursor_highlights[in_channel_ind] += max(current_highlights[out_channel_ind] * weight, -0.1)
	else:
		var out_channels = current.size[1]
		var in_channels = precursor.size[1]
		if in_channels > out_channels:
			push_error("Case in_channels > out_channels not implemented.")
		elif out_channels > in_channels:
			# We have a copy module.
			var in_channel_ind
			if in_channels==1:
				for out_channel_ind in range(out_channels):
					precursor_highlights[0] += current_highlights[out_channel_ind]
			elif current.info_code == "interleave":
				var out_channels_per_in_channel = int(out_channels / in_channels)
				for out_channel_ind in range(out_channels):
					in_channel_ind = int(out_channel_ind / out_channels_per_in_channel)
					precursor_highlights[in_channel_ind] += current_highlights[out_channel_ind]
			else:
				for out_channel_ind in range(out_channels):
					in_channel_ind = out_channel_ind % int(in_channels)
					precursor_highlights[in_channel_ind] += current_highlights[out_channel_ind]
		else:
			# We have a standard module
			for i in range(current_highlights.size()):
				precursor_highlights[i] = current_highlights[i]
