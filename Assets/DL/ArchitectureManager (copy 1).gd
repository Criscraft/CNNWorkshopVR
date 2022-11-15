extends Node2D

export var network_group_scene : PackedScene

onready var graph_edit = get_node("CustomGraphEdit")

var id_to_network_group_resource_dict = {}
var id_to_network_module_resource_dict = {}

signal set_camera_position(position)

# Called by DLManager via group.
# The method could be run in a separate thread. However, it might not be necessary because usually it is called once at startup.
# Create GroupNode
	var group_node_instance
	group_node_instance = network_group_scene.instance()
	group_node_instance.id = network_group_resource.group_id
	group_node_instance.precursors = network_group_resource.precursors
	graph_edit.get_node("MyGraphNodes").add_child(group_node_instance)
	group_node_instance.network_group_resource = network_group_resource
	
	# Wait one frame to give the boxes time to resize.
	yield(get_tree(), "idle_frame")
	graph_edit.call_deferred("arrange_nodes")
	# We have to move the camera together with its Selector node such that the Area updates its collisions.
	emit_signal("set_camera_position", Vector2.ZERO)


func create_network_group_resources(network_group_dicts):
	#var group_dicts = architecture_dict["group_dict"]
	var network_group_dict
	var network_group_resource
	
	for id in network_group_dicts:
		# Create NetworkGroupResource
		network_group_dict = network_group_dicts[id]
		network_group_resource = create_network_group_resource(network_group_dict, id)
		id_to_network_group_resource_dict[int(id)] = network_group_resource
	
	for id in id_to_network_group_resource_dict:
		network_group_resource.precursor_group_resources = []
		for id2 in network_group_resource.precursors:
			network_group_resource.precursor_group_resources.append(id_to_network_group_resource_dict[id2])
			

func create_network_group_resource(network_group_dict, id):
	var network_group_resource = NetworkGroupResource.new()
	network_group_resource.group_id = int(id)
	network_group_resource.tracker_module_group_type = NetworkGroupResource.TYPE[network_group_dict["tracker_module_group_type"]]
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
	
	for id in network_module_dicts:
		# Create NetworkGroupResource
		network_module_dict = network_module_dicts[id]
		network_module_resource = create_network_module_resource(network_module_dict, id)
		id_to_network_module_resource_dict[int(id)] = network_module_resource
	
	for id in id_to_network_module_resource_dict:
		network_module_resource.precursor_group_resources = []
		for id2 in network_module_resource.precursors:
			network_module_resource.precursor_group_resources.append(id_to_network_module_resource_dict[id2])
			

func create_network_module_resource(network_module_dict, id):
	# Create NetworkGroupResource
	var network_module_resource = NetworkModuleResource.new()
	network_module_resource.module_id = int(id)
	network_module_resource.tracker_module_type = NetworkModuleResource.TYPE[network_module_dict["tracker_module_type"]]
	# For some reason the json to dict conversion made precursors a float array. Correct that!
	var precursors = []
	for v in network_module_dict["precursors"]:
		precursors.append(int(v))
	network_module_resource.precursors = precursors
	network_module_resource.group_id = int(network_module_dict["group_id"])
	network_module_resource.label = network_module_dict["label"]
	network_module_resource.has_data = network_module_dict["has_data"]
	network_module_resource.channel_labels = network_module_dict["channel_labels"]
	network_module_resource.size = network_module_dict["size"]
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
