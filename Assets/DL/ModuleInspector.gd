extends Spatial

var network_module_resource : Resource
var lever_switch_layout_status = true
var lever_switch_activation_fv_status = true

export var network_module_details_screen_2d_scene : PackedScene
export var screen_position : int = 0

signal update_scene(resource, scene, screen_position)

func _on_Snap_Zone_has_picked_up(pickable_object):
	var module_logic = pickable_object.get_node("ModuleLogic")
	if is_instance_valid(module_logic) and \
	module_logic.network_module_resource != network_module_resource:
		network_module_resource = module_logic.network_module_resource
		update_screen()
		emit_signal("network_module_resource_updated", network_module_resource)
		
func update_screen():
	if network_module_resource == null:
		return
	if lever_switch_layout_status:
		if lever_switch_activation_fv_status:
			emit_signal("update_scene", network_module_resource, network_module_details_screen_2d_scene, screen_position)
			

func _on_lever_switch_layout_status_change(status_):
	lever_switch_layout_status = status_
	update_screen()
	

func _on_lever_switch_activation_fv_status_change(status_):
	lever_switch_activation_fv_status = status_
	update_screen()
