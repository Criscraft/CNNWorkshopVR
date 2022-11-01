extends Spatial

func _on_ModuleInspectorTable_update_scene(scene, resource, screen_position):
	var slot = $Screen.get_scene_instance().get_node("Container/Slot" + String(screen_position))
	for child in slot.get_children():
		slot.remove_child(child)
	if scene != null and resource != null:
		slot.add_child(scene.instance())
		var manager = slot.get_node("NetworkModuleDetailsScreen2D/NetworkModuleDetailsManager")
		manager.set_network_module_resource(resource)
	
