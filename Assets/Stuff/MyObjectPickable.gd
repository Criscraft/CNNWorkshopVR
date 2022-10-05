extends "res://addons/godot-xr-tools/objects/Object_pickable.gd"

func copy_pickable_to(new_parent) -> Spatial:
	# copy the pickable object
	var instance_new = duplicate(7)
	for property in get_property_list():
		if(property.usage == PROPERTY_USAGE_SCRIPT_VARIABLE):
			instance_new[property.name] = self[property.name]
	# The collision information was not copied, because the rigidbody is in static mode and the collision layer and mask are 0. We correct this manually.
	instance_new.collision_layer = original_collision_layer
	instance_new.collision_mask = original_collision_mask
	new_parent.add_child(instance_new)
	
	# apparently "var instance_new = instance_.duplicate(7)" has the same effect as the following 4 lines:
# 	var package = PackedScene.new()
#	var result = package.pack(instance_)
#	assert(result == OK)
#	var instance_new = package.instance(PackedScene.GEN_EDIT_STATE_INSTANCE)
	
	# copy the remote transform of the pickable_object
	if hold_method == HoldMethod.REMOTE_TRANSFORM:
		var remote_transform_new = _remote_transform.duplicate()
		remote_transform_new.name = remote_transform_new.name + "_copy"
		remote_transform_new.remote_path = instance_new.get_path()
		_remote_transform.get_parent().add_child(remote_transform_new)
		instance_new._remote_transform = remote_transform_new
	else:
		push_error("Grabable has no HoldMethod.REMOTE_TRANSFORM")
	return instance_new



func _on_MyPickableObject_picked_up(_pickable):
	var death_timer = get_node_or_null("DeathTimer")
	if death_timer:
		death_timer.queue_free()
