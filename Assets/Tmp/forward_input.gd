extends Control

func _gui_input(event):
	$Viewport.input(event)
	print("forwarded")

func _ready():
	$Viewport.handle_input_locally = false
	yield(get_tree(), "idle_frame")
	$Viewport.handle_input_locally = true
