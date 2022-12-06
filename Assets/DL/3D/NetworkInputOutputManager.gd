extends StaticBody

signal clear_results()
signal add_result_line(result_line)
signal add_result_node(node)
signal request_forward_pass(image_resource)
signal set_fv_image_resource(image_resource)

var current_image_resource : ImageResource = null setget set_current_image_resource
var current_fv_image_resource : ImageResource = null setget set_current_fv_image_resource
onready var test_results_button = $TestResultsButton
export var screen_width = 700

func _ready():
	var _error
	_error = connect("clear_results", $NetworkOutputScreen/Viewport2Din3DStatic.get_scene_instance(), "clear_results")
	_error = connect("add_result_line", $NetworkOutputScreen/Viewport2Din3DStatic.get_scene_instance(), "add_result_line")
	_error = connect("add_result_node", $NetworkOutputScreen/Viewport2Din3DStatic.get_scene_instance(), "add_result_node")
	_error = connect("request_forward_pass", DLManager, "on_request_forward_pass")
	_error = connect("set_fv_image_resource", DLManager, "on_set_fv_image_resource")
	_error = DLManager.connect("on_receive_test_results", self, "on_receive_test_results")
	
	
func _on_network_input_tray_picked_up(what):
	# Check if held object has image content.
	if not what.has_node("ImageLogic"):
		return
	var image_logic = what.get_node("ImageLogic")
	var image_resource = image_logic.image_resource
	emit_signal("clear_results")
	set_current_image_resource(image_resource)


func set_current_image_resource(image_resource_):
	if image_resource_.mode == ImageResource.MODE.ACTIVATION:
		emit_signal("add_result_line", "A feature map is not a valid input for the network.")
	else:
		current_image_resource = image_resource_
		emit_signal("request_forward_pass", current_image_resource.get_dict(true))


# Called by DLManager via group.
func receive_classification_results(results):
	emit_signal("clear_results")
	var class_names = results["class_names"]
	var confidence_values = results["confidence_values"]
	for i in range(class_names.size()):
		emit_signal("add_result_line", class_names[i] + " - " + confidence_values[i] + "%")


func _on_feat_vis_tray_picked_up(what):
	# Check if held object has image content.
	if not what.has_node("ImageLogic"):
		return
	var image_logic = what.get_node("ImageLogic")
	var image_resource = image_logic.image_resource
	set_current_fv_image_resource(image_resource)
	
	
func set_current_fv_image_resource(image_resource_):
	if image_resource_.mode != ImageResource.MODE.ACTIVATION:
		current_fv_image_resource = image_resource_
		emit_signal("set_fv_image_resource", current_fv_image_resource.get_dict(true))


func _on_TestResultsButton_pressed(_at):
	emit_signal("clear_results")
	emit_signal("add_result_line", "Preparing test results...")
	test_results_button.get_node("Button/CollisionShape").disabled = true
	test_results_button.get_node("InteractableAreaButton/CollisionShape").disabled = true
	DLManager.request_test_results()


func on_receive_test_results(accuracy, conv_mat):
	# Enable the button again.
	test_results_button.get_node("Button/CollisionShape").disabled = false
	test_results_button.get_node("InteractableAreaButton/CollisionShape").disabled = false
	# Write accuracy results
	emit_signal("clear_results")
	emit_signal("add_result_line", "test accuracy: " + str(float(accuracy) * 100) + "%")
	# Show image.
	var image = ImageProcessing.get_image_from_raw(conv_mat)
	var texture_rect = TextureRect.new()
	texture_rect.rect_min_size = Vector2(screen_width-50, screen_width-50)
	texture_rect.name = "ConvutionMatrix"
	var image_texture = ImageTexture.new()
	image_texture.create_from_image(image, 0)
	texture_rect.texture = image_texture
	emit_signal("add_result_node", texture_rect)
	
