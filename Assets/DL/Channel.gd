extends Control

export var line_scene : PackedScene
var margin = 10

func set_image_resource(image_resource : DLImageResource):
	$ChannelImageTile.image_resource = image_resource
	
	
func set_network_module_resource(network_module_resource : NetworkModuleResource):
	for child in get_children():
			if child.name != "ChannelImageTile":
				child.queue_free()
	if network_module_resource != null:
		add_details(network_module_resource)
		

func add_details(network_module_resource):
	if network_module_resource.kernels:
		var kernel_id = $ChannelImageTile.image_resource.channel_id % network_module_resource.kernels.size()
		var kernel = network_module_resource.kernels[kernel_id]
		var kernel_image = ImageProcessing.array_to_grayscaleimage(kernel[0])
		var kernel_texture_rect = TextureRect.new()
		kernel_texture_rect.name = "KernelImage"
		var kernel_texture = ImageTexture.new()
		kernel_texture_rect.expand = true
		kernel_texture.create_from_image(kernel_image, 0)
		kernel_texture_rect.texture = kernel_texture
		add_child(kernel_texture_rect)
		move_child(kernel_texture_rect, 0)
		kernel_texture_rect.rect_min_size = Vector2(256, 256)
		
	if network_module_resource.permutation:
		var width = 256
		var height = 256
		var channel_ind = $ChannelImageTile.image_resource.channel_id
		var perm_ind = network_module_resource.permutation[channel_ind]
		var start_pos = Vector2(width, 0.5 * height)
		var mid_pos = Vector2(0.25 * width, (perm_ind - channel_ind + 0.5) * (height + margin))
		var end_pos = Vector2(0, (perm_ind - channel_ind + 0.5) * (height + margin))
		var curve = Curve2D.new()
		curve.add_point(start_pos, Vector2(0, 0), Vector2(0, 0))
		curve.add_point(mid_pos, Vector2(0, 0), Vector2(0, 0))
		curve.add_point(end_pos, Vector2(0, 0), Vector2(0, 0))
		var line_instance = line_scene.instance()
		line_instance.set_curve(curve)
		add_child(line_instance)
			
		# Add dummy object to enforce the container to resize
		var control = ReferenceRect.new()
		control.rect_min_size = Vector2(256, 256)
		control.name = "Padding"
		add_child(control)
		move_child(control, 0)
			

func set_size_of_children(size_new):
	$ChannelImageTile.set_size_of_children(size_new)
	var channel_image_tile = $KernelImage
	if is_instance_valid(channel_image_tile):
		channel_image_tile.rect_min_size = Vector2(size_new, size_new)
