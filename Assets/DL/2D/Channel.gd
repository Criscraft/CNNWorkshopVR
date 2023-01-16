extends Control

export var weight_edit_container_scene : PackedScene = preload("res://Assets/DL/2D/WeightEditContainer.tscn")
var debouncing_timer_scene : PackedScene = preload("res://Assets/Stuff/DebouncingTimer.tscn")
var button_up_scene : PackedScene = preload("res://Assets/DL/2D/ButtonUp.tscn")
var button_down_scene : PackedScene = preload("res://Assets/DL/2D/ButtonDown.tscn")
var margin = 10
var weight_slider_height = 60
var image_height = 256
var left_weight_limit : float
var right_weight_limit : float
var channel_ind : int

signal weight_changed(weight, channel_ind, weight_ind)


func set_image_resource(image_resource : ImageResource):
	$ChannelImageTile.image_resource = image_resource
	channel_ind = $ChannelImageTile.image_resource.channel_id
	
	
func clear_details():
	for child in $CenterContainer/Details.get_children():
		remove_child(child)
		child.queue_free()


func draw_PFModule_kernels(kernel):
	var kernel_image = ImageProcessing.array_to_grayscaleimage(kernel[0])
	var kernel_texture_rect = TextureRect.new()
	kernel_texture_rect.name = "KernelImage"
	var kernel_texture = ImageTexture.new()
	kernel_texture_rect.expand = true
	kernel_texture.create_from_image(kernel_image, 0)
	kernel_texture_rect.texture = kernel_texture
	$CenterContainer/Details.add_child(kernel_texture_rect)
	#$Details.move_child(kernel_texture_rect, 0)
	kernel_texture_rect.rect_min_size = Vector2(256, 256)
	
	
func create_weights(weights, weight_limit, weight_name):
	var weight_edit = weight_edit_container_scene.instance()
	$CenterContainer/Details.add_child(weight_edit)
	weight_edit.create_weights(weights, weight_limit, weight_name, self)


func create_permutation_buttons(indices, index_name):
	var button_up = button_up_scene.instance()
	$CenterContainer/Details.add_child(button_up)
	button_up.connect("pressed", self, "on_permutation_button_up_pressed", [index_name])
	var button_down = button_down_scene.instance()
	$CenterContainer/Details.add_child(button_down)
	button_down.connect("pressed", self, "on_permutation_button_down_pressed", [index_name])


func on_permutation_button_up_pressed(index_name):
	emit_signal("weight_changed", -1, channel_ind, null, index_name)
	
	
func on_permutation_button_down_pressed(index_name):
	emit_signal("weight_changed", 1, channel_ind, null, index_name)


func set_size_of_children(size_new):
	$ChannelImageTile.set_size_of_children(size_new)


# Called by WeightEditContainer via signal. 
# Also called by on_button_up_pressed and on_button_down_pressed
func on_weight_changed(weight, weight_ind, weight_name):
	emit_signal("weight_changed", weight, channel_ind, weight_ind, weight_name)
	
