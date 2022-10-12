extends Control


func _input(event):
	print("event_mini")
	
func _gui_input(event):
	print("gui_event_mini")

func _ready():
	set_process_input(true)
