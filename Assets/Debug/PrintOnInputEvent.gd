extends Node
export var prefix = ""

func _ready():
	 #Input.use_accumulated_input = false
	pass

func _input(event):
	print("input " + prefix)
	
func _unhandled_input(event):
	print("unhandled_input " + prefix)

func _gui_input(event):
	print("gui_input " + prefix)
