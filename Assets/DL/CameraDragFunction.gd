extends Control

var is_dragging : bool = false
var is_cursor_on_vp : bool = false

onready var camera = $Camera2D

# For some reason, the viewport changes the subclassed InputEventMouseMotionEmulated back into InputEventMouseMotion event. I do not understand, why
# For some reason, the viewport only propagates the input event on Control nodes that collide with the mouse pointer.
func _gui_input(event):
	if not is_cursor_on_vp:
		is_dragging = false
		return
	if event is InputEventMouseMotion and is_dragging:
		camera.translate(-event.relative)
	else:
		if event is InputEventMouseButton:
			is_dragging = event.is_pressed()

func _on_pointer_entered():
	is_cursor_on_vp = true

func _on_pointer_exited():
	is_cursor_on_vp = false
