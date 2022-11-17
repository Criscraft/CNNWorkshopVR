extends Control

export var slider_scene : PackedScene


func set_initial_weights(weights, left_limit, right_limit):
	var slider
	for i in range(len(weights)):
		slider = slider_scene.instance()
		slider.rect_min_size.y = int((256 - 10 * len(weights)) / len(weights))
		slider.min_value = left_limit
		slider.max_value = right_limit
		slider.value = weights[i]
		add_child(slider)
		slider.connect("value_changed", get_parent(), "on_weight_changed", [i])


func set_color(child_ind, color):
	var slider = get_child(child_ind)
	slider.self_modulate = color
	
	
func set_value(child_ind, value):
	var slider = get_child(child_ind)
	slider.value = value
