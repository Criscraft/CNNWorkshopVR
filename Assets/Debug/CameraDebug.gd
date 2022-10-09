
extends Camera

onready var Player = get_parent()

export (float, 0.0, 1.0) var acceleration = 1.0
export (float, 0.0, 0.0, 1.0) var deceleration = 0.1
export var max_speed = Vector3(1.0, 1.0, 1.0)
export var local = true
export var forward_action = "ui_up"
export var backward_action = "ui_down"
export var left_action = "ui_left"
export var right_action = "ui_right"
export var up_action = "ui_page_up"
export var down_action = "ui_page_down"

var _direction = Vector3(0.0, 0.0, 0.0)
var _speed = Vector3(0.0, 0.0, 0.0)

## Increase this value to give a slower turn speed
const CAMERA_TURN_SPEED = 200

var mouse_captured_mode : bool = true

func _ready():
	Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
	## Tell Godot that we want to handle input
	set_process_input(true)

# Called on each frame to update the pickup
func _process(delta):
	
	_update_movement(delta)
	
	if Input.is_action_just_pressed("ESC"):
		if mouse_captured_mode:
			Input.mouse_mode = Input.MOUSE_MODE_VISIBLE
		else:
			Input.mouse_mode = Input.MOUSE_MODE_CAPTURED
		mouse_captured_mode = not mouse_captured_mode
		
		
func _update_movement(delta):
	
	if Input.is_action_pressed(forward_action):
		_direction.z = -1
	elif Input.is_action_pressed(backward_action):
		_direction.z = 1
	elif not Input.is_action_pressed(forward_action) and not Input.is_action_pressed(backward_action):
		_direction.z = 0

	if Input.is_action_pressed(left_action):
		_direction.x = -1
	elif Input.is_action_pressed(right_action):
		_direction.x = 1
	elif not Input.is_action_pressed(left_action) and not Input.is_action_pressed(right_action):
		_direction.x = 0
		
	if Input.is_action_pressed(up_action):
		_direction.y = 1
	if Input.is_action_pressed(down_action):
		_direction.y = -1
	elif not Input.is_action_pressed(up_action) and not Input.is_action_pressed(down_action):
		_direction.y = 0
	
	var offset = max_speed * acceleration * _direction
	
	_speed.x = clamp(_speed.x + offset.x, -max_speed.x, max_speed.x)
	_speed.y = clamp(_speed.y + offset.y, -max_speed.y, max_speed.y)
	_speed.z = clamp(_speed.z + offset.z, -max_speed.z, max_speed.z)
	
	# Apply deceleration if no input
	if _direction.x == 0:
		_speed.x *= (1.0 - deceleration)
	if _direction.y == 0:
		_speed.y *= (1.0 - deceleration)
	if _direction.z == 0:
		_speed.z *= (1.0 - deceleration)

	if local:
		translate(_speed * delta)
	else:
		global_translate(_speed * delta)
			
			

func look_updown_rotation(rotation = 0):
	"""
	Returns a new Vector3 which contains only the x direction
	We'll use this vector to compute the final 3D rotation later
	"""
	var toReturn = self.get_rotation() + Vector3(rotation, 0, 0)

	##
	## We don't want the player to be able to bend over backwards
	## neither to be able to look under their arse.
	## Here we'll clamp the vertical look to 90° up and down
	toReturn.x = clamp(toReturn.x, PI / -2, PI / 2)

	return toReturn

func look_leftright_rotation(rotation = 0):
	"""
	Returns a new Vector3 which contains only the y direction
	We'll use this vector to compute the final 3D rotation later
	"""
	return Player.get_rotation() + Vector3(0, rotation, 0)

func _input(event):
	"""
	First person camera controls
	"""
	##
	## We'll only process mouse motion events
	if not mouse_captured_mode or not event is InputEventMouseMotion:
		return

	##
	## We'll use the parent node "Player" to set our left-right rotation
	## This prevents us from adding the x-rotation to the y-rotation
	## which would result in a kind of flight-simulator camera
	Player.set_rotation(look_leftright_rotation(event.relative.x / -CAMERA_TURN_SPEED))

	##
	## Now we can simply set our y-rotation for the camera, and let godot
	## handle the transformation of both together
	self.set_rotation(look_updown_rotation(event.relative.y / -CAMERA_TURN_SPEED))

func _enter_tree():
	"""
	Hide the mouse when we start
	"""
	Input.set_mouse_mode(Input.MOUSE_MODE_HIDDEN)

func _leave_tree():
	"""
	Show the mouse when we leave
	"""
	Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
