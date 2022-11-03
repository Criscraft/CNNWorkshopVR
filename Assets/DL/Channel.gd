extends Control


export var image_tile_scene : PackedScene


func set_image_resource(image_resource : DLImageResource):
	$ChannelImageTile.image_resource = image_resource
	
	
func set_network_module_resource(network_module_resource : NetworkModuleResource):
	for child in get_children():
			if child.name != "ChannelImageTile":
				child.queue_free()
	if network_module_resource != null:
		add_details(network_module_resource)
		

func add_details(_network_module_resource):
	pass

func set_size_of_children(size_new):
	$ChannelImageTile.set_size_of_children(size_new)
	# TODO: Resize detail nodes.
