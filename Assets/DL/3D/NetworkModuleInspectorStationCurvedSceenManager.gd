extends Spatial

export var network_module_details_screen2D_scene : PackedScene = preload("res://Assets/DL/2D/NetworkModuleDetailsScreen2D.tscn")
export var image_tile_scene : PackedScene = preload("res://Assets/DL/2D/ImageTile.tscn")
export var n_slots : int = 5
export var radius_inspector_tables : float = 0.7
export var angle_range_inspector_tables : int = 130
export var network_module_inspector_table_scene : PackedScene = preload("res://Assets/DL/3D/NetworkModuleInspectorTable.tscn")
export var pickable_module_scene : PackedScene = preload("res://Assets/DL/3D/PickableNetworkModule.tscn")
onready var container = $Screen.get_scene_instance().get_node("Container")
onready var tables = $Tables


func _ready():
	var angle_range_rad = deg2rad(angle_range_inspector_tables)
	var angle_step_size = angle_range_rad / (n_slots - 1)
	var angles = []
	for i in range(n_slots):
		angles.append(i * angle_step_size - 0.5 * angle_range_rad)
		
	var slot
	var table
	var pos_x
	var pos_z
	var pos_glob
	var rad_glob
	for i in range(n_slots):
		# Compute position
		pos_x = - radius_inspector_tables * sin(angles[i]) # we go from positive x to negative x (left to right)
		pos_z = radius_inspector_tables * cos(angles[i])
		# Add screen slot
		slot = HBoxContainer.new()
		slot.alignment = BoxContainer.ALIGN_CENTER
		container.add_child(slot)
		# Add inspector table
		table = network_module_inspector_table_scene.instance()
		table.screen_slot = slot
		tables.add_child(table)
		pos_glob = tables.to_global(Vector3(pos_x, 0, pos_z))
		rad_glob = tables.to_global(Vector3(2*pos_x, 0, 2*pos_z))
		table.look_at_from_position(pos_glob, rad_glob, Vector3.UP)
	for i in range(n_slots):
		table = tables.get_child(i)
		if i > 0:
			table.previous_screen_slot = tables.get_child(i-1).screen_slot
		if i < n_slots-1:
			table.next_screen_slot = tables.get_child(i+1).screen_slot
		


func get_slot(screen_id):
	return container.get_child(screen_id)
	

class MyCustomSorter:
	static func sort_ascending(a, b):
		if a.module_id < b.module_id:
			return true
		return false
		
		
# Called via group on_fastfill_pressed by network module provider
func on_fastfill_pressed(group_id : int):
	if group_id < 0:
		return
	
	# Remove the old modules.
	for i in range(n_slots):
		var snap_zone = tables.get_child(i).get_node("ModuleTray/Snap_Zone")
		snap_zone.destroy_held_item()
	
	# Add the new modules.
	var network_module_resources = ArchitectureManager.group_id_to_network_module_resources[group_id]
	network_module_resources.sort_custom(MyCustomSorter, "sort_ascending")
	for i in range(network_module_resources.size()):
		if i == n_slots:
			break
		var network_module_resource = network_module_resources[i]
		var snap_zone = tables.get_child(i).get_node("ModuleTray/Snap_Zone")
		var pickable_module = create_portable_module_at(network_module_resource)
		pickable_module.global_transform = snap_zone.global_transform


func create_portable_module_at(network_module_resource):
	var pickable_module = pickable_module_scene.instance()
	var spawned_items_node = get_tree().get_root().get_node("Staging/SpawnedItems")
	spawned_items_node.add_child(pickable_module)
	pickable_module.get_node("ModuleLogic").network_module_resource = network_module_resource
	return pickable_module
