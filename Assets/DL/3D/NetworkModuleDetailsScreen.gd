extends Spatial

func _ready():
	# The module details manager should update its content when the detail screen selects a module.
	$Viewport2Din3D/Viewport/NetworkModuleDetailsScreen2D/NetworkModuleDetailsManager.add_to_group("on_network_module_selected_by_detail_screen")
