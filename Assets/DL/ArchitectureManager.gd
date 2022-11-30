extends Node

export var group_id_to_network_group_resource_dict = {}
export var module_id_to_network_module_resource_dict = {}
export var group_id_to_network_module_resources = {}
onready var channel_highlighting_script = preload("res://Assets/DL/ChannelHighlighting.gd")
var channel_highlighting

signal created_network_group_resources
signal created_network_module_resources

func _ready():
	channel_highlighting = Node.new()
	channel_highlighting.name = "ChannelHighlighting"
	channel_highlighting.set_script(channel_highlighting_script)
	channel_highlighting.add_to_group("on_image_selected")
	add_child(channel_highlighting)

func create_network_group_resources(network_group_dicts):
	var network_group_dict
	var network_group_resource
	
	for id in network_group_dicts:
		# Create NetworkGroupResource
		network_group_dict = network_group_dicts[id]
		network_group_resource = create_network_group_resource(network_group_dict, id)
		group_id_to_network_group_resource_dict[int(id)] = network_group_resource
	
	for id in group_id_to_network_group_resource_dict:
		network_group_resource = group_id_to_network_group_resource_dict[id]
		network_group_resource.precursor_group_resources = []
		for id2 in network_group_resource.precursors:
			network_group_resource.precursor_group_resources.append(group_id_to_network_group_resource_dict[id2])
			
	emit_signal("created_network_group_resources")
	

func create_network_group_resource(network_group_dict, id):
	var network_group_resource = NetworkGroupResource.new()
	network_group_resource.group_id = int(id)
	# For some reason the json to dict conversion made precursors a float array. Correct that!
	var precursors = []
	for v in network_group_dict["precursors"]:
		precursors.append(int(v))
	network_group_resource.precursors = precursors
	network_group_resource.label = network_group_dict["label"]
	return network_group_resource


func create_network_module_resources(network_module_dicts):
	var network_module_dict
	var network_module_resource
	var precursor
	
	# Create network module resources
	for id in network_module_dicts:
		network_module_dict = network_module_dicts[id]
		network_module_resource = create_network_module_resource(network_module_dict, id)
		module_id_to_network_module_resource_dict[int(id)] = network_module_resource
		
		if not network_module_resource.group_id in group_id_to_network_module_resources:
			group_id_to_network_module_resources[network_module_resource.group_id] = []
		group_id_to_network_module_resources[network_module_resource.group_id].append(network_module_resource)
		
	
	# Add precursors and successors to network module resources
	for id in module_id_to_network_module_resource_dict:
		network_module_resource = module_id_to_network_module_resource_dict[id]
		for id2 in network_module_resource.precursors:
			precursor = module_id_to_network_module_resource_dict[id2]
			network_module_resource.precursor_module_resources.append(precursor)
			precursor.successor_module_resources.append(network_module_resource)
		network_module_resource.create_input_mapping()
	
	channel_highlighting.initialize()
	emit_signal("created_network_module_resources")
			

func create_network_module_resource(network_module_dict, id):
	# Create NetworkGroupResource
	var network_module_resource = NetworkModuleResource.new()
	network_module_resource.module_id = int(id)
	# For some reason the json to dict conversion made precursors a float array. Correct that!
	var precursors = []
	for v in network_module_dict["precursors"]:
		precursors.append(int(v))
	network_module_resource.precursors = precursors
	network_module_resource.group_id = int(network_module_dict["group_id"])
	network_module_resource.label = network_module_dict["label"]
	network_module_resource.channel_labels = network_module_dict["channel_labels"]
	network_module_resource.size = network_module_dict["size"]
	network_module_resource.info_code = network_module_dict["info_code"]
	if "weights" in network_module_dict:
		network_module_resource.weights = network_module_dict["weights"]
		network_module_resource.weights_min = network_module_dict["weights_min"]
		network_module_resource.weights_max = network_module_dict["weights_max"]
	if "permutation" in network_module_dict:
		network_module_resource.permutation = network_module_dict["permutation"]
	if "kernels" in network_module_dict:
		network_module_resource.kernels = network_module_dict["kernels"]
	if "padding" in network_module_dict:
		network_module_resource.padding = network_module_dict["padding"]
	return network_module_resource
