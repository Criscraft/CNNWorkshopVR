extends Spatial

export var module_resource : Resource setget set_module_resource

func set_module_resource(module_resource_):
	module_resource = module_resource_
	# Update the displayed label. If this node is being duplicated and the copy is not yet added to the scene tree, we skip.
	if is_inside_tree():
		update_label()
	
func update_label() -> void:
	$Label.get_scene_instance().get_node("Label").text = module_resource.label
