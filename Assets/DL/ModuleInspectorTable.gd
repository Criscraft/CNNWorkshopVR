extends Spatial

var network_module_resource : Resource setget set_network_module_resource
var image_resource : Resource setget set_image_resource
var lever_switch_layout_details = true
var lever_switch_activation_fv_status = true

export var screen_position : int = 0

signal change_network_module_resource(network_module_resource, screen_position, details_layout, feature_visualization_mode)
signal change_image_resource(image_resource, screen_position)
signal change_details_layout(screen_position, mode)
signal toggle_module_notes(screen_position)

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
	# Note that when lever_switch_activation_fv_status is true we want to show activations, not feature visualizations
	emit_signal("change_network_module_resource", network_module_resource, screen_position, lever_switch_layout_details, not lever_switch_activation_fv_status)
	

func set_image_resource(image_resource_):
	if image_resource == image_resource_:
		return
		
	image_resource = image_resource_
	emit_signal("change_image_resource", image_resource, screen_position)


func _on_lever_switch_layout_status_change(status_):
	lever_switch_layout_details = status_
	# Note that when lever_switch_layout_details is true we want to see details
	emit_signal("change_details_layout", screen_position, lever_switch_layout_details)
	

func _on_lever_switch_activation_fv_status_change(status_):
	lever_switch_activation_fv_status = status_
	# Note that when lever_switch_activation_fv_status is true we want to show activations, not feature visualizations
	print("You changed lever_switch_activation_fv_status but this is not yet implemented")


func _on_ToggleLayout_pressed(_button):
	emit_signal("toggle_module_notes", screen_position)
