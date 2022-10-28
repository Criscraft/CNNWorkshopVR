extends "res://Assets/Stuff/MyPickableObject.gd"


func add_duplicate_to(new_parent) -> Spatial:
	var scene = load("res://Assets/DL/PickableModule.tscn")
	var instance_new = scene.instance()
	instance_new.global_transform = global_transform
	new_parent.add_child(instance_new)
	# set image resource (only possible after being added to scene tree)
	instance_new.get_node("ModuleLogic").network_module_resource = get_node("ModuleLogic").network_module_resource
	return instance_new
