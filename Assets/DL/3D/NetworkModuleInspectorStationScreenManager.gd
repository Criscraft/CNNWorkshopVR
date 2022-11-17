extends Spatial

export var network_module_details_screen2D_scene : PackedScene = preload("res://Assets/DL/2D/NetworkModuleDetailsScreen2D.tscn")
export var image_tile_scene : PackedScene = preload("res://Assets/DL/2D/ImageTile.tscn")

func _on_ModuleInspectorTable_set_details_layout(screen_position, mode):
	var slot = $Screen.get_scene_instance().get_node("Container/Slot" + String(screen_position))
	if slot.get_child_count() > 0:
		var manager = slot.get_node("NetworkModuleDetailsScreen2D/NetworkModuleDetailsManager")
		manager.details_layout = mode


func _on_ModuleInspectorTable_set_network_module_resource(network_module_resource, screen_position, details_layout, feature_visualization_mode):
	var slot = $Screen.get_scene_instance().get_node("Container/Slot" + String(screen_position))
	for child in slot.get_children():
		child.queue_free()
	if network_module_resource != null:
		slot.add_child(network_module_details_screen2D_scene.instance())
		var manager = slot.get_node("NetworkModuleDetailsScreen2D/NetworkModuleDetailsManager")
		manager.feature_visualization_mode = feature_visualization_mode
		manager.details_layout = details_layout
		manager.network_module_resource = network_module_resource
		# By default the ModuleNotesPanel should be invisible to save space.
		manager.module_notes_panel_visibility = false


func _on_ModuleInspectorTable_toggle_module_notes(screen_position):
	var slot = $Screen.get_scene_instance().get_node("Container/Slot" + String(screen_position))
	if slot.get_child_count() > 0:
		var manager = slot.get_node("NetworkModuleDetailsScreen2D/NetworkModuleDetailsManager")
		manager.module_notes_panel_visibility = not manager.module_notes_panel_visibility


func _on_ModuleInspectorTable_set_image_resource(image_resource, screen_position):
	var slot = $Screen.get_scene_instance().get_node("Container/Slot" + String(screen_position))
	for child in slot.get_children():
		child.queue_free()
	if image_resource != null:
		var image_tile = image_tile_scene.instance()
		slot.add_child(image_tile)
		image_tile.image_resource = image_resource
		image_tile.set_size_of_children(512)


func _on_ModuleInspectorTable_set_feature_visualization_mode(screen_position, mode):
	var slot = $Screen.get_scene_instance().get_node("Container/Slot" + String(screen_position))
	if slot.get_child_count() > 0:
		var manager = slot.get_node("NetworkModuleDetailsScreen2D/NetworkModuleDetailsManager")
		manager.feature_visualization_mode = mode
