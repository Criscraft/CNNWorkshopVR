extends Spatial

onready var screen = $Screen

func _on_ModuleInspectorTable_update_scene(resource, scene, screen_position):
	screen.scene = network_module_details_screen_2d_scene
	var manager = screen.get_scene_instance().get_node("NetworkModuleDetailsManager")
	manager.set_network_module_resource(resource)
			
