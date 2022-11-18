extends Spatial

export var network_module_details_screen2D_scene : PackedScene = preload("res://Assets/DL/2D/NetworkModuleDetailsScreen2D.tscn")
export var image_tile_scene : PackedScene = preload("res://Assets/DL/2D/ImageTile.tscn")
export var n_slots : int = 3
export var network_module_inspector_table_scene : PackedScene = preload("res://Assets/DL/3D/NetworkModuleInspectorTable.tscn")
export var network_module_inspector_table_width : float = 0.36


func _ready():
	var container = $Screen.get_scene_instance().get_node("Container")
	var slot
	var table
	var center = network_module_inspector_table_width * (n_slots - 1) / 2
	for i in range(n_slots):
		# Add screen slot
		slot = HBoxContainer.new()
		slot.alignment = BoxContainer.ALIGN_CENTER
		container.add_child(slot)
		# Add inspector table
		table = network_module_inspector_table_scene.instance()
		table.translation.x = i * network_module_inspector_table_width - center
		table.screen_slot = slot
		add_child(table)


func get_slot(screen_id):
	return $Container.get_child(screen_id)
	
