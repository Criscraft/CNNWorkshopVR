class_name MyXRTSnapZone
extends XRTSnapZone

export var can_ranged_grab: bool = true
export var copies_on_pick_up: bool = true

func drop_object() -> void:
	if not is_instance_valid(picked_up_object):
		return
	
	# let go of this object
	picked_up_object.let_go(Vector3.ZERO, Vector3.ZERO)
	
	emit_signal("has_dropped")
	
	if copies_on_pick_up:
		# Make a deep copy of the object.
		var picked_up_object_new = picked_up_object.copy_pickable_to(picked_up_object.get_parent())
		picked_up_object = null
	else:
		picked_up_object = null
		emit_signal("highlight_updated", self, true)

