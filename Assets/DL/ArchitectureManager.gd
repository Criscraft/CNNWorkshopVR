extends Node2D

export var network_group_scene : PackedScene

# Called by DLManager via group.
func receive_architecture(architecture_dict):
	var group_dict = architecture_dict["group_dict"]
	var value
	var network_group_resource
	var group_node_instance
	var precursors
	
	for id in group_dict:
		# Create NetworkGroupResource
		value = group_dict[id]
		network_group_resource = NetworkGroupResource.new()
		network_group_resource.id = int(id)
		network_group_resource.tracker_module_group_type = NetworkGroupResource.TYPE[value["tracker_module_group_type"]]
		# For some reason the json to dict conversion made precursors a float array. Correct that!
		precursors = []
		for v in value["precursors"]:
			precursors.append(int(v))
		network_group_resource.precursors = precursors
		network_group_resource.label = value["label"]
		# Create GroupNode
		group_node_instance = network_group_scene.instance()
		$CustomGraphEdit.add_child(group_node_instance)
		group_node_instance.network_group_resource = network_group_resource
		
	$CustomGraphEdit.arrange_nodes()
		
		
