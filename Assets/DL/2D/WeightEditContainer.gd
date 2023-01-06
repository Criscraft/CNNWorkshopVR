extends Control

export var slider_scene : PackedScene = preload("res://Assets/DL/2D/WeightEditScrollBar.tscn")
export var line_scene : PackedScene = preload("res://Assets/GraphHandling/DrawnPath2D.tscn")
export var slider_width = 60
export var line_width = 256
export var image_height = 256
export var margin = 10
export var n_steps = 5
	
func create_weights(weights, weight_limit, weight_name, channel_instance, vertical=true):
	var n_weights = len(weights)
	for i in range(n_weights):
		var slider = slider_scene.instance()
		slider.rect_min_size.y = int((image_height - 10 * n_weights) / n_weights)
		slider.step = (weight_limit[1] - weight_limit[0]) / (n_steps - 1)
		slider.min_value = weight_limit[0]
		slider.max_value = weight_limit[1] + slider.step
		slider.page = slider.step
		slider.value = weights[i]
		var color = ImageProcessing.get_colormap_color(weights[i], [slider.min_value, slider.max_value])
		slider.get("custom_styles/scroll/StyleBoxFlat").bg_color = color
		add_child(slider)
		var _err = slider.connect("value_changed", channel_instance, "on_weight_changed", [i, weight_name])
		_err = slider.connect("value_changed", self, "on_weight_changed", [i])


# Called by this node via signal.
func on_weight_changed(weight, weight_ind):
	var slider = get_child(weight_ind)
	var color = ImageProcessing.get_colormap_color(weight, [slider.min_value, slider.max_value])
	slider.get("custom_styles/scroll/StyleBoxFlat").bg_color = color


func get_line(from : Vector2, to : Vector2):
	var curve = Curve2D.new()
	curve.add_point(from, Vector2(0, 0), Vector2(0, 0))
	curve.add_point(to, Vector2(0, 0), Vector2(0, 0))
	var line_instance = line_scene.instance()
	line_instance.set_curve(curve)
	return line_instance
