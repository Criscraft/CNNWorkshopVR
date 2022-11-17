extends Spatial

var network_module_resource : Resource setget set_network_module_resource
var image_resource : Resource setget set_image_resource
var layout_details_mode = true
var feature_visualization_mode = false

export var screen_position : int = 0

signal set_network_module_resource(network_module_resource, screen_position, details_layout, feature_visualization_mode)
signal set_image_resource(image_resource, screen_position)
signal set_details_layout(screen_position, mode)
signal toggle_module_notes(screen_position)
signal set_feature_visualization_mode(screen_position, mode)

func _on_Snap_Zone_has_picked_up(pickable_object):
	var module_logic = pickable_object.get_node_or_null("ModuleLogic")
	if is_instance_valid(module_logic):
		# A pickable module was placed into the snap tray
		set_network_module_resource(module_logic.network_module_resource)
		return
	
	var image_logic = pickable_object.get_node_or_null("ImageLogic")
	if is_instance_valid(image_logic):
		# A pickable image was placed into the snap tray
		set_image_resource(image_logic.image_resource)
		return


func _on_Snap_Zone_has_dropped():
	set_network_module_resource(null)
	set_image_resource(null)
	
	
func set_network_module_resource(network_module_resource_):
	if network_module_resource == network_module_resource_:
		return
		
	network_module_resource = network_module_resource_
	# Note that when lever_switch_layout_status is true we want to see details
	# Note that when feature_visualization_mode is false we want to show activations
	emit_signal("set_network_module_resource", network_module_resource, screen_position, layout_details_mode, feature_visualization_mode)
	

func set_image_resource(image_resource_):
	if image_resource == image_resource_:
		return
		
	image_resource = image_resource_
	emit_signal("set_image_resource", image_resource, screen_position)


func _on_lever_switch_layout_status_change(status_):
	layout_details_mode = status_
	# Note that when lever_switch_layout_details is true we want to see details
	emit_signal("set_details_layout", screen_position, layout_details_mode)
	

func _on_lever_switch_activation_fv_status_change(status_):
	feature_visualization_mode = not status_
	# Note that when status_ is true we want to show activations, not feature visualizations
	emit_signal("set_feature_visualization_mode", screen_position, feature_visualization_mode)


func _on_ToggleLayout_pressed(_button):
	emit_signal("toggle_module_notes", screen_position)
