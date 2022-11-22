extends Control

export var line_scene : PackedScene
export var weight_edit_scene : PackedScene
var margin = 10
var left_weight_limit : float
var right_weight_limit : float
var channel_ind : int

signal weight_changed(weight, channel_ind, weight_ind)


func set_image_resource(image_resource : ImageResource):
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
		channel_ind = $ChannelImageTile.image_resource.channel_id
		var perm_ind = network_module_resource.permutation[channel_ind]
		draw_channel_connection(channel_ind, perm_ind)
		add_dummy_rect()
		
	if network_module_resource.weights:
		var in_channels_per_group = network_module_resource.weights[0].size()
		var out_channels = network_module_resource.size[1]
		var in_channels = out_channels
		if network_module_resource.precursor_module_resources.size() > 0:
			in_channels = network_module_resource.precursor_module_resources[0].size[1]
		var n_groups = in_channels / in_channels_per_group
		var out_channels_per_group = out_channels / n_groups
		channel_ind = $ChannelImageTile.image_resource.channel_id
		var group_ind = int(channel_ind / out_channels_per_group)
		var channel_offset = group_ind * in_channels_per_group
		
		var color
		var weights = []
		for weight in network_module_resource.weights[channel_ind]:
			weights.append(weight[0][0])
		left_weight_limit = -1.0
		right_weight_limit = 1.0
		"""
		if network_module_resource.weights_min < 0 and network_module_resource.weights_max < 0:
			left_weight_limit = network_module_resource.weights_min
		elif network_module_resource.weights_min > 0 and network_module_resource.weights_max > 0:
			right_weight_limit = network_module_resource.weights_max
		else:
			left_weight_limit = - max(-network_module_resource.weights_min, network_module_resource.weights_max)
			right_weight_limit = - left_weight_limit
		"""
		
		# Draw weight edit UI
		var weight_edit = weight_edit_scene.instance()
		add_child(weight_edit)
		move_child(weight_edit, 0)
		weight_edit.set_initial_weights(weights, left_weight_limit, right_weight_limit)
		
		# Draw channel connections and colorize
		
		var y_offset
		for i in range(in_channels_per_group):
			color = get_weight_color(weights[i])
			weight_edit.set_color(i, color)
			y_offset =  (i + 0.5) / in_channels_per_group
			draw_channel_connection(channel_ind, channel_offset + i, color, y_offset)
			
		# Draw dummy rect such that the channel connections have enough space to be shown
		add_dummy_rect()
			
			
func get_weight_color(weight):
	var color : Color
	if weight < 0:
		color = Color(0, 0, weight / (left_weight_limit + 1e-6))
	else:
		color = Color(weight / (right_weight_limit + 1e-6), 0, 0)
	return color
	
	
func draw_channel_connection(this_channel_id, precursor_channel_id, color=null, relative_shift_right=0.5):
	var width = 256
	var height = 256
	var this_global_position_y = this_channel_id * (height + margin)
	var precursor_global_position_y = precursor_channel_id * (height + margin)
	var relative_precursor_position_y = precursor_global_position_y - this_global_position_y
	var start_pos = Vector2(width, relative_shift_right * height)
	var mid_pos = Vector2(0.25 * width, relative_precursor_position_y + 0.5 * height)
	var end_pos = Vector2(0, relative_precursor_position_y + 0.5 * height)
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


# Called by WeightEditContainer via signal.
func on_weight_changed(weight, weight_ind):
	var color = get_weight_color(weight)
	$WeightEditContainer.set_color(weight_ind, color)
	emit_signal("weight_changed", weight, channel_ind, weight_ind)
	
