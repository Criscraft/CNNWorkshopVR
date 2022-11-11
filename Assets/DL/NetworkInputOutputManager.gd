extends StaticBody

signal clear_results()
signal add_result_line(result_line)
signal request_forward_pass(image_resource)

var current_image_resource : DLImageResource = null setget set_current_image_resource

func _ready():
	var _error
	_error = connect("clear_results", $NetworkOutputScreen/Viewport2Din3DStatic.get_scene_instance(), "clear_results")
	_error = connect("add_result_line", $NetworkOutputScreen/Viewport2Din3DStatic.get_scene_instance(), "add_result_line")
	_error = connect("request_forward_pass", DLManager, "on_request_forward_pass")
	
	
func _on_network_input_tray_picked_up(what):
	# Check if held object has image content.
	if not what.has_node("ImageLogic"):
		return
	var image_logic = what.get_node("ImageLogic")
	var image_resource = image_logic.image_resource
	emit_signal("clear_results")
	if image_resource.mode == DLImageResource.MODE.ACTIVATION:
		emit_signal("add_result_line", "A feature map is not a valid input for the network.")
	else:
		set_current_image_resource(image_resource)


func set_current_image_resource(new_value):
	current_image_resource = new_value
	emit_signal("request_forward_pass", current_image_resource)


# Called by DLManager via group.
func receive_classification_results(results):
	emit_signal("clear_results")
	var class_names = results["class_names"]
	var confidence_values = results["confidence_values"]
	for i in range(class_names.size()):
		emit_signal("add_result_line", class_names[i] + " - " + confidence_values[i] + "%")


func _on_feat_vis_tray_picked_up(what):
	pass # Replace with function body.
