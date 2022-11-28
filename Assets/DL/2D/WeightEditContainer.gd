extends Control

export var slider_scene : PackedScene = preload("res://Assets/DL/2D/WeightEditScrollBar.tscn")
export var line_scene : PackedScene = preload("res://Assets/GraphHandling/DrawnPath2D.tscn")
export var slider_width = 60
export var line_width = 256
export var image_height = 256
export var margin = 10
var left_limit
var right_limit

func set_initial_weights(input_indices, weights, left_limit_, right_limit_):
	left_limit = left_limit_
	right_limit = right_limit_
	var slider
	var local_from
	var local_to
	var line
	for i in range(len(weights)):
		slider = slider_scene.instance()
		slider.rect_min_size.y = int((image_height - 10 * len(weights)) / len(weights))
		slider.min_value = left_limit
		slider.max_value = right_limit
		slider.value = weights[i][0][0] # It is a 1x1 kernel.
		add_child(slider)
		slider.connect("value_changed", get_parent(), "on_weight_changed", [i])
	
	yield(get_tree(), "idle_frame")
	# Draw lines
	for i in range(len(weights)):
		slider = get_child(i)
		local_from = Vector2(0.0, slider_width/2)
		local_to = Vector2(-line_width, -slider.rect_global_position.y + (input_indices[i] + 0.5) * (image_height + margin))
		line = get_line(local_from, local_to)
		line.name = "Line"
		slider.add_child(line)
		set_weight(i, weights[i][0][0])


func set_weight(child_ind, weight):
	var color = ImageProcessing.get_weight_color(weight)
	var slider = get_child(child_ind)
	slider.get("custom_styles/scroll/StyleBoxFlat").bg_color = color
	slider.get_node("Line").set_color(color)
	
	
func set_value(child_ind, value):
	var slider = get_child(child_ind)
	slider.value = value


func get_line(from : Vector2, to : Vector2):
	var curve = Curve2D.new()
	curve.add_point(from, Vector2(0, 0), Vector2(0, 0))
	curve.add_point(to, Vector2(0, 0), Vector2(0, 0))
	var line_instance = line_scene.instance()
	line_instance.set_curve(curve)
	return line_instance
