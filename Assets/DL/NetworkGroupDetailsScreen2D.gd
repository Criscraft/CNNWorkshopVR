extends PanelContainer

var network_group_resource


func network_group_selected_by_overvie_screen(network_group_resource_):
	if network_group_resource != network_group_resource_:
		network_group_resource = network_group_resource_
		update()
		

func update():
	$Label.text = network_group_resource.label
