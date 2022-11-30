extends Spatial

export var pickable_module_scene : PackedScene = preload("res://Assets/DL/3D/PickableNetworkModule.tscn")
onready var snap_zone = $ModuleProviderTray/Snap_Zone

# Called by NetworkGroupSelector via group method
func network_module_selected_by_detail_screen(network_module):
	snap_zone.destroy_held_item()
	create_portable_module(network_module.network_module_resource)
	
	
func create_portable_module(network_module_resource):
	var pickable_module = pickable_module_scene.instance()
	pickable_module.global_transform = snap_zone.global_transform
	var spawned_items_node = get_tree().get_root().get_node("Staging/SpawnedItems")
	spawned_items_node.add_child(pickable_module)
	pickable_module.get_node("ModuleLogic").network_module_resource = network_module_resource


func _on_fastfill_pressed(_at=null):
	if snap_zone.picked_up_object != null:
		var module_logic = snap_zone.picked_up_object.get_node_or_null("ModuleLogic")
		if module_logic != null:
			var group_id = module_logic.network_module_resource.group_id
			get_tree().call_group("on_fastfill_pressed", "on_fastfill_pressed", group_id)
