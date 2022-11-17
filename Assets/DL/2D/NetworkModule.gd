extends "res://Assets/GraphHandling/MyGraphNode.gd"

var network_module_resource setget set_network_module_resource

func set_network_module_resource(new):
	network_module_resource = new
	set_text(network_module_resource.label)
