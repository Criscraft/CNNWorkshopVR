extends Spatial

var pickable_fv_settings_scene : PackedScene = preload("res://Assets/DL/3D/PickableFVSettings.tscn")

func _ready():
	var new_instance = pickable_fv_settings_scene.instance()
	var fv_settings_resource = FVSettingsResource.new()
	new_instance.get_node("FVSettings").fv_settings_resource = fv_settings_resource
	var spawned_items_node = get_tree().get_root().get_node("Staging/SpawnedItems")
	spawned_items_node.add_child(new_instance)
	new_instance.global_transform = $MySnapTray/Snap_Zone.global_transform
