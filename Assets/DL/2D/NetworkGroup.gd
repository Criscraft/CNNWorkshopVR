extends "res://Assets/GraphHandling/MyGraphNode.gd"

var network_group_resource setget set_network_group_resource

func set_network_group_resource(new):
	network_group_resource = new
	set_text(network_group_resource.label)
