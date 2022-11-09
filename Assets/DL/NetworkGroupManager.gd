extends Node2D

export var network_module_scene : PackedScene
onready var graph_edit = $CustomGraphEdit
var group_id_to_module_dict = {}
var current_group_id : int = -1

# Called on selection by network overview screen via group.
# The method could be run in a separate thread.
func network_group_selected_by_overview_screen(network_group):
	var group_id = network_group.network_group_resource.group_id
	
	if current_group_id == group_id:
		return
	
	# Remove old module nodes from graph edit.
	var my_graph_nodes = graph_edit.get_node("MyGraphNodes")
	for child in my_graph_nodes.get_children():
		my_graph_nodes.remove_child(child)
		
	# Add new module nodes to graph edit.
	for node in group_id_to_module_dict[group_id]:
		my_graph_nodes.add_child(node)
		
	current_group_id = group_id
	
	# Wait one frame to give the boxes time to resize.
	yield(get_tree(), "idle_frame")
	graph_edit.call_deferred("arrange_nodes")
	
	

func receive_architecture(architecture_dict):
	# Collect all group ids:
	for id in architecture_dict["group_dict"]:
		group_id_to_module_dict[int(id)] = []
	
	# Create network module nodes
	var module_dict = architecture_dict["module_dict"]
	var value
	var network_module_resource
	var group_node_instance
	var precursors
	for id in module_dict:
		# Create NetworkGroupResource
		value = module_dict[id]
		network_module_resource = NetworkModuleResource.new()
		network_module_resource.module_id = int(id)
		network_module_resource.tracker_module_type = NetworkModuleResource.TYPE[value["tracker_module_type"]]
		# For some reason the json to dict conversion made precursors a float array. Correct that!
		precursors = []
		for v in value["precursors"]:
			precursors.append(int(v))
		network_module_resource.precursors = precursors
		network_module_resource.group_id = int(value["group_id"])
		network_module_resource.label = value["label"]
		network_module_resource.has_data = value["has_data"]
		network_module_resource.channel_labels = value["channel_labels"]
		network_module_resource.size = value["size"]
		if "weights" in value:
			network_module_resource.weights = value["weights"]
			network_module_resource.weights_min = value["weights_min"]
			network_module_resource.weights_max = value["weights_max"]
		if "permutation" in value:
			network_module_resource.permutation = value["permutation"]
		if "kernels" in value:
			network_module_resource.kernels = value["kernels"]
		if "padding" in value:
			network_module_resource.padding = value["padding"]
		
		# Create module node
		group_node_instance = network_module_scene.instance()
		group_node_instance.id = network_module_resource.module_id
		group_node_instance.precursors = network_module_resource.precursors
		group_node_instance.network_module_resource = network_module_resource
		
		# Append the module node to our dictionary
		group_id_to_module_dict[network_module_resource.group_id].append(group_node_instance)
	
