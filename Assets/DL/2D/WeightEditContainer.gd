extends Control

export var slider_scene : PackedScene = preload("res://Assets/DL/2D/WeightEditScrollBar.tscn")
export var line_scene : PackedScene = preload("res://Assets/GraphHandling/DrawnPath2D.tscn")
export var slider_width = 60
export var line_width = 256
export var image_height = 256
export var margin = 10
var left_limit
var right_limit

func set_initial_weights(weights, left_limit_, right_limit_, first_input_channel):
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
		slider.value = weights[i]
		add_child(slider)
		slider.connect("value_changed", get_parent(), "on_weight_changed", [i])
	
	yield(get_tree(), "idle_frame")
	# Draw lines
	for i in range(len(weights)):
		slider = get_child(i)
		local_from = Vector2(0.0, slider_width/2)
		local_to = Vector2(-line_width, -slider.rect_global_position.y + (first_input_channel + 0.5 + i) * (image_height + margin))
		line = get_line(local_from, local_to, get_weight_color(weights[i]))
		line.name = "Line"
		slider.add_child(line)


func set_weight(child_ind, weight):
	var color = get_weight_color(weight)
	var slider = get_child(child_ind)
	slider.get("custom_styles/scroll/StyleBoxFlat").bg_color = color
	slider.get_node("Line").set_color(color)
	
	
func set_value(child_ind, value):
	var slider = get_child(child_ind)
	slider.value = value


func get_line(from : Vector2, to : Vector2, color=null):
	var curve = Curve2D.new()
	curve.add_point(from, Vector2(0, 0), Vector2(0, 0))
	curve.add_point(to, Vector2(0, 0), Vector2(0, 0))
	var line_instance = line_scene.instance()
	line_instance.set_curve(curve)
	if color != null:
		line_instance.set_color(color)
	return line_instance
	
	
func get_weight_color(weight):
	var color : Color
	if weight < 0:
		color = Color(0, 0, weight / (left_limit + 1e-6))
	else:
		color = Color(weight / (right_limit + 1e-6), 0, 0)
	return color
