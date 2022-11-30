
extends "res://Assets/DesktopPlayer/camera.gd"

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

# Called on each frame to update the pickup
func _process(delta):
	# ._process(delta) #This is called automatically
	_update_movement(delta)
	
	
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
		player.translate(_speed * delta)
	else:
		player.global_translate(_speed * delta)
		
		
