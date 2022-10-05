class_name Function_Pickup_Desktop
extends Spatial

## Signal emitted when the pickup picks something up
signal has_picked_up(what)

## Signal emitted when the pickup drops something
signal has_dropped

# Constant for worst-case grab distance
const MAX_GRAB_DISTANCE2: float = 1000000.0

## Hand node, which creates a translation for the held item.
export (NodePath) var _hand

## Grab distance
export var grab_distance: float = 0.3 setget _set_grab_distance

## Grab collision mask
export (int, LAYERS_3D_PHYSICS) var grab_collision_mask = 1 setget _set_grab_collision_mask

## Enable ranged-grab
export var ranged_enable: bool = true

## Ranged-grab distance
export var ranged_distance: float = 5.0 setget _set_ranged_distance

## Ranged-grab angle
export (float, 0.0, 45.0) var ranged_angle: float = 5.0 setget _set_ranged_angle

## Ranged-grab collision mask
export (int, LAYERS_3D_PHYSICS) var ranged_collision_mask = 1 setget _set_ranged_collision_mask

## Throw impulse factor
export var impulse_factor: float = 1.0

## Throw velocity avetaging
export var velocity_samples: int = 5


# Public fields
var closest_object: Spatial = null
var picked_up_object: Spatial = null
var picked_up_ranged: bool = false

# Private fields
var _object_in_grab_area = Array()
var _object_in_ranged_area = Array()
var _velocity_averager = VelocityAverager.new(velocity_samples)
var _grab_area: Area
var _grab_collision: CollisionShape
var _ranged_area: Area
var _ranged_collision: CollisionShape

# Called when the node enters the scene tree for the first time.
func _ready():
	
	_hand = get_node(_hand)
	
	var _errorcode # irrelevant
	
	# Skip if running in the editor
	if Engine.editor_hint:
		return

	# Create the grab collision shape
	_grab_collision = CollisionShape.new()
	_grab_collision.set_name("GrabCollisionShape")
	_grab_collision.shape = SphereShape.new()
	_grab_collision.shape.radius = grab_distance

	# Create the grab area
	_grab_area = Area.new()
	_grab_area.set_name("GrabArea")
	_grab_area.collision_layer = 0
	_grab_area.collision_mask = grab_collision_mask
	_grab_area.add_child(_grab_collision)
	_errorcode = _grab_area.connect("area_entered", self, "_on_grab_entered")
	_errorcode = _grab_area.connect("body_entered", self, "_on_grab_entered")
	_errorcode = _grab_area.connect("area_exited", self, "_on_grab_exited")
	_errorcode = _grab_area.connect("body_exited", self, "_on_grab_exited")
	add_child(_grab_area)

	# Create the ranged collision shape
	_ranged_collision = CollisionShape.new()
	_ranged_collision.set_name("RangedCollisionShape")
	_ranged_collision.shape = CylinderShape.new()
	_ranged_collision.transform.basis = Basis(Vector3.RIGHT, PI/2)

	# Create the ranged area
	_ranged_area = Area.new()
	_ranged_area.set_name("RangedArea")
	_ranged_area.collision_layer = 0
	_ranged_area.collision_mask = ranged_collision_mask
	_ranged_area.add_child(_ranged_collision)
	_errorcode = _ranged_area.connect("area_entered", self, "_on_ranged_entered")
	_errorcode = _ranged_area.connect("body_entered", self, "_on_ranged_entered")
	_errorcode = _ranged_area.connect("area_exited", self, "_on_ranged_exited")
	_errorcode = _ranged_area.connect("body_exited", self, "_on_ranged_exited")
	add_child(_ranged_area)

	# Update the colliders
	_update_colliders()


# Called on each frame to update the pickup
func _process(delta):
	# Calculate average velocity
	if is_instance_valid(picked_up_object) and picked_up_object.is_picked_up():
		# Average velocity of picked up object
		_velocity_averager.add_transform(delta, picked_up_object.global_transform)
	else:
		# Average velocity of this pickup
		_velocity_averager.add_transform(delta, global_transform)

	_update_closest_object()
	
	# Process Input. Ignore the press_to_hold property of the clostes_object. In Desktop we never want to hold the key.
	if Input.is_action_just_pressed("pickup"):
		if is_instance_valid(picked_up_object):
			drop_object()
		elif is_instance_valid(closest_object):
			_pick_up_object(closest_object)
	if Input.is_action_just_pressed("pickup_action"):
		if is_instance_valid(picked_up_object) and picked_up_object.has_method("action"):
			picked_up_object.action()


# Called when the grab distance has been modified
func _set_grab_distance(var new_value: float) -> void:
	grab_distance = new_value
	if is_inside_tree():
		_update_colliders()


# Called when the grab collision mask has been modified
func _set_grab_collision_mask(var new_value: int) -> void:
	grab_collision_mask = new_value
	if is_inside_tree() and _grab_collision:
		_grab_collision.collision_mask = new_value


# Called when the ranged-grab distance has been modified
func _set_ranged_distance(var new_value: float) -> void:
	ranged_distance = new_value
	if is_inside_tree():
		_update_colliders()


# Called when the ranged-grab angle has been modified
func _set_ranged_angle(var new_value: float) -> void:
	ranged_angle = new_value
	if is_inside_tree():
		_update_colliders()


# Called when the ranged-grab collision mask has been modified
func _set_ranged_collision_mask(var new_value: int) -> void:
	ranged_collision_mask = new_value
	if is_inside_tree() and _ranged_collision:
		_ranged_collision.collision_mask = new_value


# Update the colliders geometry
func _update_colliders() -> void:
	# Update the grab sphere
	if _grab_collision:
		_grab_collision.shape.radius = grab_distance

	# Update the ranged-grab cylinder
	if _ranged_collision:
		_ranged_collision.shape.radius = tan(deg2rad(ranged_angle)) * ranged_distance
		_ranged_collision.shape.height = ranged_distance
		_ranged_collision.transform.origin.z = -ranged_distance * 0.5


# Called when an object enters the grab sphere
func _on_grab_entered(target: Spatial) -> void:
	# reject objects which don't support picking up
	if not target.has_method('pick_up'):
		return

	# ignore objects already known
	if _object_in_grab_area.find(target) >= 0:
		return

	# Add to the list of objects in grab area
	_object_in_grab_area.push_back(target)


# Called when an object enters the ranged-grab cylinder
func _on_ranged_entered(target: Spatial) -> void:
	# reject objects which don't support picking up rangedly
	if not 'can_ranged_grab' in target or not target.can_ranged_grab:
		return

	# ignore objects already known
	if _object_in_ranged_area.find(target) >= 0:
		return

	# Add to the list of objects in grab area
	_object_in_ranged_area.push_back(target)


# Called when an object exits the grab sphere
func _on_grab_exited(target: Spatial) -> void:
	_object_in_grab_area.erase(target)


# Called when an object exits the ranged-grab cylinder
func _on_ranged_exited(target: Spatial) -> void:
	_object_in_ranged_area.erase(target)


# Update the closest object field with the best choice of grab
func _update_closest_object() -> void:
	# Find the closest object we can pickup
	var new_closest_obj: Spatial = null
	if not picked_up_object:
		# Find the closest in grab area
		new_closest_obj = _get_closest_grab()
		if not new_closest_obj and ranged_enable:
			# Find closest in ranged area
			new_closest_obj = _get_closest_ranged()

	# Skip if no change
	if closest_object == new_closest_obj:
		return

	# remove highlight on old object
	if is_instance_valid(closest_object):
		closest_object.decrease_is_closest()

	# add highlight to new object
	closest_object = new_closest_obj
	if is_instance_valid(closest_object):
		closest_object.increase_is_closest()


# Find the pickable object closest to our hand's grab location
func _get_closest_grab() -> Spatial:
	var new_closest_obj: Spatial = null
	var new_closest_distance := MAX_GRAB_DISTANCE2
	for o in _object_in_grab_area:
		# skip objects that can not be picked up
		if not o.can_pick_up(self):
			continue

		# Save if this object is closer than the current best
		var distance_squared := global_transform.origin.distance_squared_to(o.global_transform.origin)
		if distance_squared < new_closest_distance:
			new_closest_obj = o
			new_closest_distance = distance_squared

	# Return best object
	return new_closest_obj


# Find the rangedly-pickable object closest to our hand's pointing direction
func _get_closest_ranged() -> Spatial:
	var new_closest_obj: Spatial = null
	var new_closest_angle_dp := cos(deg2rad(ranged_angle))
	var hand_forwards := -global_transform.basis.z
	for o in _object_in_ranged_area:
		# skip objects that can not be picked up
		if not o.can_pick_up(self):
			continue

		# Save if this object is closer than the current best
		var object_direction: Vector3 = o.global_transform.origin - global_transform.origin
		object_direction = object_direction.normalized()
		var angle_dp := hand_forwards.dot(object_direction)
		if angle_dp > new_closest_angle_dp:
			new_closest_obj = o
			new_closest_angle_dp = angle_dp

	# Return best object
	return new_closest_obj


func drop_object() -> void:
	if not is_instance_valid(picked_up_object):
		return

	# Remove ranged collision shape from held object
#	var ranged_collision_shape = picked_up_object.get_node("RangedCollisionShape")
#	if is_instance_valid(ranged_collision_shape):
#		ranged_collision_shape.queue_free()

	# let go of this object
	picked_up_object.let_go(
		- global_transform.basis.z * impulse_factor,
		Vector3.ZERO)
	picked_up_object = null
	emit_signal("has_dropped")


func _pick_up_object(target: Spatial) -> void:
	# check if already holding an object
	if is_instance_valid(picked_up_object):
		# skip if holding the target object
		if picked_up_object == target:
			return
		# holding something else? drop it
		drop_object()

	# skip if target null or freed
	if not is_instance_valid(target):
		return

	# Handle snap-zone
	var snap := target as XRTSnapZone
	if snap:
		target = snap.picked_up_object
		snap.drop_object()

	# Pick up our target. Note, target may do instant drop_and_free
	picked_up_ranged = not _object_in_grab_area.has(target)
	picked_up_object = target
	target.pick_up(self, null)
	
	# Hack to make sure, that the target is at the hand position. Assumes, that the grabbable object uses reparenting. Tha _hand and this Function_Pickup node have to have the same parent, because the remote transform stores a relative path to the held object.
	if target.hold_method == target.HoldMethod.REMOTE_TRANSFORM:
		var source = get_node("PickupRemoteTransform")
		remove_child(source)
		_hand.add_child(source)
		source.set_owner(_hand)
	else:
		push_error("Tried to grab a grabable that has HoldMethod.REMOTE_TRANSFORM")
	
	# Add collision shape to held object. The player can put the object into a snap tray from a distance.
#	_ranged_collision = CollisionShape.new()
#	_ranged_collision.set_name("RangedCollisionShape")
#	_ranged_collision.shape = CylinderShape.new()
#	_ranged_collision.shape.radius = 0.2
#	_ranged_collision.shape.height = 6
#	_ranged_collision.transform = _ranged_collision.transform.translated(Vector3(0, 0, -3))
#	_ranged_collision.transform.basis = _ranged_collision.transform.basis * Basis(Vector3.RIGHT, PI/2)
	
	#target.add_child(_ranged_collision)

	# If object picked up then emit signal
	if is_instance_valid(picked_up_object):
		emit_signal("has_picked_up", picked_up_object)
		
