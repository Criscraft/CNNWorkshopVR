class_name MyXRTSnapZone
extends XRTSnapZone

export var can_ranged_grab: bool = true
export var copies_on_pick_up: bool = true


# Called on each frame to update the pickup
func _process(_delta):
	if not copies_on_pick_up and is_instance_valid(picked_up_object):
		return

	for o in _object_in_grab_area:
		# skip objects that can not be picked up
		if not o.can_pick_up(self):
			continue

		# pick up our target
		_pick_up_object(o)
		return
		
		
# Pickup Method: Drop the currently picked up object. Create a copy of the current object.
func drop_object() -> void:
	if not is_instance_valid(picked_up_object):
		return

	# Make a deep copy of the object.
	var picked_up_object_new = null
	if copies_on_pick_up:
		picked_up_object_new = picked_up_object.copy_pickable_to(picked_up_object.get_parent())
	
	# let go of this object
	picked_up_object.let_go(Vector3.ZERO, Vector3.ZERO)
	
	emit_signal("has_dropped")
	
	if copies_on_pick_up:
		picked_up_object = picked_up_object_new
	else:
		picked_up_object = null
		emit_signal("highlight_updated", self, true)


# Pick up the specified object
func _pick_up_object(target: Spatial) -> void:
	# check if already holding an object
	if not copies_on_pick_up and is_instance_valid(picked_up_object):
		# skip if holding the target object
		if picked_up_object == target:
			return
		# holding something else? drop it
		drop_object()
		
	if not copies_on_pick_up and is_instance_valid(picked_up_object):
		picked_up_object.queue_free()
		picked_up_object = null

	# skip if target null or freed
	if not is_instance_valid(target):
		return

	# Pick up our target. Note, target may do instant drop_and_free
	picked_up_object = target
	target.pick_up(self, null)

	# If object picked up then emit signal
	if is_instance_valid(picked_up_object):
		emit_signal("has_picked_up", picked_up_object)
		emit_signal("highlight_updated", self, false)
