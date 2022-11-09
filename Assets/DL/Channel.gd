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
		draw_channel_connection(channel_ind, perm_ind)
		add_dummy_rect()
		
	if network_module_resource.weights:
		var n_channels = network_module_resource.weights.size()
		var group_size = network_module_resource.weights[0].size()
		var channel_ind = $ChannelImageTile.image_resource.channel_id
		var group = int(channel_ind / group_size)
		var group_position = channel_ind % group_size
		var color
		var weight
		var left_limit = 0.0
		var right_limit = 0.0
		if network_module_resource.weights_min < 0 and network_module_resource.weights_max < 0:
			left_limit = network_module_resource.weights_min
		elif network_module_resource.weights_min > 0 and network_module_resource.weights_max > 0:
			right_limit = network_module_resource.weights_max
		else:
			left_limit = - max(-network_module_resource.weights_min, network_module_resource.weights_max)
			right_limit = - left_limit
		# Draw channel connections
		for i in range(group_size):
			weight = network_module_resource.weights[channel_ind][group_position][0][0]
			if weight < 0:
				color = Color(0, 0, weight / left_limit)
			else:
				color = Color(weight / right_limit, 0, 0)
			draw_channel_connection(channel_ind, group * group_size + i, color)
		add_dummy_rect()
			
func draw_channel_connection(channel_id, child_channel_id, color=null):
	var width = 256
	var height = 256
	var start_pos = Vector2(width, 0.5 * height)
	var mid_pos = Vector2(0.25 * width, (child_channel_id - channel_id + 0.5) * (height + margin))
	var end_pos = Vector2(0, (child_channel_id - channel_id + 0.5) * (height + margin))
	var curve = Curve2D.new()
	curve.add_point(start_pos, Vector2(0, 0), Vector2(0, 0))
	curve.add_point(mid_pos, Vector2(0, 0), Vector2(0, 0))
	curve.add_point(end_pos, Vector2(0, 0), Vector2(0, 0))
	var line_instance = line_scene.instance()
	line_instance.set_curve(curve)
	if color != null:
		line_instance.set_color(color)
	add_child(line_instance)


func add_dummy_rect():
	# Add dummy object to enforce the container to resize
	var width = 256
	var height = 256
	var control = ReferenceRect.new()
	control.rect_min_size = Vector2(width, height)
	control.name = "Padding"
	add_child(control)
	move_child(control, 0)

func set_size_of_children(size_new):
	$ChannelImageTile.set_size_of_children(size_new)
	var channel_image_tile = get_node_or_null("KernelImage")
	if is_instance_valid(channel_image_tile):
		channel_image_tile.rect_min_size = Vector2(size_new, size_new)
