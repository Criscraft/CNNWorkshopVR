extends Spatial

var network_module_resource : Resource
var lever_switch_layout_status = true
var lever_switch_activation_fv_status = true

export var screen_position : int = 0

signal change_network_module_resource(network_module_resource, screen_position, details_layout, feature_visualization_mode)
signal change_details_layout(screen_position, mode)
signal toggle_module_notes(screen_position)

func _on_Snap_Zone_has_picked_up(pickable_object):
	var module_logic = pickable_object.get_node("ModuleLogic")
	if is_instance_valid(module_logic):
		set_network_module_resource(module_logic.network_module_resource)


func _on_Snap_Zone_has_dropped():
	set_network_module_resource(null)
	
	
func set_network_module_resource(network_module_resource_):
	if network_module_resource == network_module_resource_:
		return
		
	network_module_resource = network_module_resource_
	# Note that when lever_switch_layout_status is true we want to have no details
	# Note that when lever_switch_activation_fv_status is true we want to show activations, not feature visualizations
	emit_signal("change_network_module_resource", network_module_resource, screen_position, not lever_switch_layout_status, not lever_switch_activation_fv_status)
			

func _on_lever_switch_layout_status_change(status_):
	lever_switch_layout_status = status_
	# Note that when lever_switch_layout_status is true we want to have no details
	emit_signal("change_details_layout", screen_position, not lever_switch_layout_status)
	

func _on_lever_switch_activation_fv_status_change(status_):
	lever_switch_activation_fv_status = status_
	# Note that when lever_switch_activation_fv_status is true we want to show activations, not feature visualizations
	print("You changed lever_switch_activation_fv_status but this is not yet implemented")


func _on_ToggleLayout_pressed(_button):
	emit_signal("toggle_module_notes", screen_position)
