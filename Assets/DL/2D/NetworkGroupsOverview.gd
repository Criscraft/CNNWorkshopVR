extends Node2D

export var network_group_scene : PackedScene

onready var graph_edit = $CustomGraphEdit

signal set_camera_position(position)

func _ready():
	var _error
	_error = ArchitectureManager.connect("created_network_group_resources", self, "create_group_nodes")
	
	
func create_group_nodes():
	# The method could be run in a separate thread. However, it might not be necessary because usually it is called once at startup.
	var group_node_instance
	var network_group_resource
	
	for group_id in ArchitectureManager.id_to_network_group_resource_dict:
		
		network_group_resource = ArchitectureManager.id_to_network_group_resource_dict[group_id]
		
		group_node_instance = network_group_scene.instance()
		group_node_instance.id = network_group_resource.group_id
		group_node_instance.precursors = network_group_resource.precursors
		group_node_instance.network_group_resource = network_group_resource
		
		graph_edit.get_node("MyGraphNodes").add_child(group_node_instance)
		
	# Wait one frame to give the boxes time to resize.
	yield(get_tree(), "idle_frame")
	graph_edit.call_deferred("arrange_nodes")
	# We have to move the camera together with its Selector node such that the Area updates its collisions.
	emit_signal("set_camera_position", Vector2.ZERO)

