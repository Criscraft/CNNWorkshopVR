extends PanelContainer

export var network_group_resource : Resource setget set_network_group_resource

func set_network_group_resource(network_group_resource_):
	network_group_resource = network_group_resource_
	$MarginContainer/Titel.text = network_group_resource.label
