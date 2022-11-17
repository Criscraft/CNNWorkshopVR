extends Spatial

## Signal for hinge moved
signal hinge_moved(status_)

## Hinge minimum limit, points up
export var hinge_limit_min := -45.0 setget _set_hinge_limit_min
## Hinge maximum limit, points down
export var hinge_limit_max := 45.0 setget _set_hinge_limit_max
## Hinge position
export var hinge_position := 0.0 setget _set_hinge_position

# Hinge values in radians
onready var _hinge_limit_min_rad := deg2rad(hinge_limit_min)
onready var _hinge_limit_max_rad := deg2rad(hinge_limit_max)
onready var _hinge_position_rad := deg2rad(hinge_position)

export var status : bool = true setget set_status


# Called when the node enters the scene tree for the first time.
func _ready():
	# Set default position
	set_status(status)


func switch_status():
	set_status(not status)


# Toggle hingMove the hinge to the specified position
func set_status(status_ : bool) -> void:
	status = status_
	if status: 
		move_hinge(_hinge_limit_max_rad)
	else:
		move_hinge(_hinge_limit_min_rad)
		
	# Emit the moved signal
	emit_signal("hinge_moved", status)
	

func move_hinge(position):
	# Skip if the position has not changed
	if position == _hinge_position_rad:
		return

	# Update the current positon
	_hinge_position_rad = position
	hinge_position = rad2deg(position)

	# Update the transform
	transform.basis = Basis(Vector3(_hinge_position_rad, 0, 0))


# Called when hinge_limit_min is set externally
func _set_hinge_limit_min(var value: float) -> void:
	hinge_limit_min = value
	_hinge_limit_min_rad = deg2rad(value)


# Called when hinge_limit_max is set externally
func _set_hinge_limit_max(var value: float) -> void:
	hinge_limit_max = value
	_hinge_limit_max_rad = deg2rad(value)


# Called when hinge_position is set externally
func _set_hinge_position(var value: float) -> void:
	hinge_position = value
	_hinge_position_rad = deg2rad(value)
	if is_inside_tree():
		move_hinge(_hinge_position_rad)


func _on_InteractableHandle_pointer_pressed(_at):
	switch_status()
