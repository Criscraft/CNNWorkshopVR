extends Control


func _gui_input(event):
	if event is InputEventMouseButton and event.button_index == BUTTON_LEFT:
		if event.is_pressed():
			$Sprite.position = event.position
