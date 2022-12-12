extends Control

export var line_scene : PackedScene = preload("res://Assets/Stuff/TextLine.tscn")
export var weight_edit_container_scene : PackedScene = preload("res://Assets/DL/2D/WeightEditContainer.tscn")
var margin = 10
var weight_slider_height = 60
var image_height = 256
var left_weight_limit : float
var right_weight_limit : float

signal weight_changed(weight, channel_ind, weight_ind)


func set_image_resource(image_resource : ImageResource):
	$ChannelImageTile.image_resource = image_resource
	
	
func set_network_module_resource(network_module_resource : NetworkModuleResource):
	for child in get_children():
		if child.name != "ChannelImageTile":
			remove_child(child)
			child.queue_free()
	if network_module_resource != null:
		add_details(network_module_resource)
		

func add_details(network_module_resource):
	var channel_ind = $ChannelImageTile.image_resource.channel_id
	if network_module_resource.kernels:
		# We have a nxn convolution with n>1
		# Draw the kernels.
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
		return
		
	if network_module_resource.weights:
		# We have a 1x1 convolution
		# Draw color coded edges.
		var weight_edit = weight_edit_container_scene.instance()
		add_child(weight_edit)
		move_child(weight_edit, 0)
		weight_edit.set_initial_weights(network_module_resource.input_mapping[channel_ind], network_module_resource.weights[channel_ind], network_module_resource.weight_limit_min, network_module_resource.weight_limit_max)
		# Draw dummy rect such that the channel connections have enough space to be shown
		add_dummy_rect()
		return
		
	if network_module_resource.precursor_module_resources:
		var in_channels = int(network_module_resource.precursor_module_resources[0].size[1])
		var out_channels = int(network_module_resource.size[1])
		# If we have a copy module or if we have a permutation module we draw edges. 
		if in_channels < out_channels or network_module_resource.permutation:
			var in_channel_ind = network_module_resource.input_mapping[channel_ind][0]
			draw_channel_connection(channel_ind, in_channel_ind)
			# Draw dummy rect such that the channel connections have enough space to be shown
			add_dummy_rect()
		
	
func draw_channel_connection(this_channel_id, precursor_channel_id):
	var width = image_height
	var this_global_position_y = this_channel_id * (image_height + margin)
	var precursor_global_position_y = precursor_channel_id * (image_height + margin)
	var relative_precursor_position_y = precursor_global_position_y - this_global_position_y
	var start_pos = Vector2(width, 0.5 * width)
	var mid_pos = Vector2(0.25 * width, relative_precursor_position_y + 0.5 * image_height)
	var end_pos = Vector2(0, relative_precursor_position_y + 0.5 * image_height)
	var curve = Curve2D.new()
	curve.add_point(start_pos, Vector2(0, 0), Vector2(0, 0))
	curve.add_point(mid_pos, Vector2(0, 0), Vector2(0, 0))
	curve.add_point(end_pos, Vector2(0, 0), Vector2(0, 0))
	var line_instance = line_scene.instance()
	line_instance.set_curve(curve)
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
	$WeightEditContainer.set_weight(weight_ind, weight)
	var channel_ind = $ChannelImageTile.image_resource.channel_id
	emit_signal("weight_changed", weight, channel_ind, weight_ind)
	
