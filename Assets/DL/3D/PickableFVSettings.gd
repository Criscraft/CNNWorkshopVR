extends "res://Assets/VRInteraction/MyPickableObject.gd"


func add_duplicate_to(new_parent) -> Spatial:
	var scene = load(filename)
	var instance_new = scene.instance()
	instance_new.global_transform = global_transform
	new_parent.add_child(instance_new)
	# set image resource (only possible after being added to scene tree)
	instance_new.get_node("FVSettings").fv_settings_resource = get_node("FVSettings").fv_settings_resource
	return instance_new
