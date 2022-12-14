extends Node2D

export var network_module_scene : PackedScene = preload("res://Assets/DL/2D/NetworkModule.tscn")
onready var graph_edit = $CustomGraphEdit
var group_id_to_network_module_instances = {}
var current_group_id : int = -1

func _ready():
	var _error
	_error = ArchitectureManager.connect("created_network_module_resources", self, "create_module_nodes")

# Called on selection by network overview screen via group.
# The method could be run in a separate thread.
func network_group_selected_by_overview_screen(network_group):
	var group_id = network_group.network_group_resource.group_id
	
	if current_group_id == group_id:
		return
	
	current_group_id = group_id
	
	# Remove old module nodes from graph edit.
	var my_graph_nodes = graph_edit.get_node("MyGraphNodes")
	for child in my_graph_nodes.get_children():
		my_graph_nodes.remove_child(child)
		
	if group_id in group_id_to_network_module_instances:
		# Add new module nodes to graph edit.
		for node in group_id_to_network_module_instances[group_id]:
			my_graph_nodes.add_child(node)
	
	# Wait one frame to give the boxes time to resize.
	yield(get_tree(), "idle_frame")
	graph_edit.call_deferred("arrange_nodes")
	

func create_module_nodes():
	var module_node_instance
	var network_module_resource
	
	for module_id in ArchitectureManager.module_id_to_network_module_resource_dict:
		
		network_module_resource = ArchitectureManager.module_id_to_network_module_resource_dict[module_id]
		
		module_node_instance = network_module_scene.instance()
		module_node_instance.id = network_module_resource.module_id
		module_node_instance.precursors = network_module_resource.precursors.duplicate()
		module_node_instance.network_module_resource = network_module_resource
		
		if not network_module_resource.group_id in group_id_to_network_module_instances:
			group_id_to_network_module_instances[network_module_resource.group_id] = []
		group_id_to_network_module_instances[network_module_resource.group_id].append(module_node_instance)
		
	# Cleaning step: remove precursors that belong to a different group_id
	var module_ids = []
	for group_id in group_id_to_network_module_instances:
		module_ids = []
		for module_instance in group_id_to_network_module_instances[group_id]:
			module_ids.append(module_instance.id)
		for module_instance in group_id_to_network_module_instances[group_id]:
			for precursor in module_instance.precursors.duplicate():
				if not precursor in module_ids:
					module_instance.precursors.erase(precursor)
