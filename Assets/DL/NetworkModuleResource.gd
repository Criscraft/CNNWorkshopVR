extends Resource
class_name NetworkModuleResource

# Guaranteed to be complete and meaningful
export var module_id : int = -1
export var group_id : int = -1
export var label : String = ""
export var precursors : Array = []
export var precursor_module_resources : Array = []
export var successor_module_resources : Array = []
# Implemented tags: ignore_highlight, draw_edges
export var tags : Array = []
export var size : Array = []

# Optional properties
export var data : Dictionary = {}
var weight_names = ["grouped_conv_weight", "sparse_conv_weight_selection", "sparse_conv_weight_group", "blend_weight", "weight_per_channel", "indices"]
	
	
func _init():
	# Initialize arrays as they can sometimes share memory otherwise.
	precursors = []
	precursor_module_resources = []
	successor_module_resources = []
	tags = []
	size = []
	data = {}


func get_dict():
	# Export only data that can be used to identify the module
	var out = {
		"module_id" : module_id,
		"group_id" : group_id,
	}
	return out


#func create_input_mapping():
#	if not precursors:
#		return
#
#	# We assume that the input mapping is identical for all precursors.
#	var precursor = precursor_module_resources[0]
#	var out_channels = size[1]
#	var in_channels = precursor.size[1]
#	input_mapping = []
#	for _i in range(out_channels):
#		input_mapping.append([])
#
#	if permutation:
#		for i in range(permutation.size()):
#			input_mapping[i].append(permutation[i])
#	elif weights:
#		var in_channels_per_group = weights[0].size()
#		var n_groups = in_channels / in_channels_per_group
#		var out_channels_per_group = out_channels / n_groups
#		var group_ind
#		var in_channel_ind
#		for out_channel_ind in range(weights.size()):
#			group_ind = int(out_channel_ind / out_channels_per_group)
#			for groupmember_ind in range(in_channels_per_group):
#				in_channel_ind = group_ind * in_channels_per_group + groupmember_ind
#				input_mapping[out_channel_ind].append(in_channel_ind)
#	else:
#		if in_channels > out_channels:
#			push_error("Case in_channels > out_channels not implemented.")
#		elif out_channels > in_channels:
#			# We have a copy module.
#			var in_channel_ind
#			if in_channels==1:
#				for out_channel_ind in range(out_channels):
#					input_mapping[out_channel_ind].append(0)
#			elif info_code == "interleave":
#				var out_channels_per_in_channel = int(out_channels / in_channels)
#				for out_channel_ind in range(out_channels):
#					in_channel_ind = int(out_channel_ind / out_channels_per_in_channel)
#					input_mapping[out_channel_ind].append(in_channel_ind)
#			else:
#				for out_channel_ind in range(out_channels):
#					in_channel_ind = out_channel_ind % int(in_channels)
#					input_mapping[out_channel_ind].append(in_channel_ind)
#		else:
#			# We have a standard module
#			for i in range(out_channels):
#				input_mapping[i].append(i)
