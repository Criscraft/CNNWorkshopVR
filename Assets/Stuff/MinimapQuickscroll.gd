extends Control

signal minimap_quickscroll(position)
	
func _gui_input(event):
	if event is InputEventMouseButton and event.is_pressed() and event.button_index == BUTTON_LEFT:
		# The event position seems to be recalculated to match the local coordinate system. 
		# We want to set the viewport camera using global coordinates. 
		# So we correct this by adding the position.
		emit_signal("minimap_quickscroll", event.position + get_rect().position)
