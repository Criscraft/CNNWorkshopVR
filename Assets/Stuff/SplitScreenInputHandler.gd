extends VBoxContainer

export var view_port_main_path : NodePath
onready var view_port_main = get_node(view_port_main_path)
export var view_port_minimap_path : NodePath
onready var view_port_minimap = get_node(view_port_minimap_path)

func _gui_input(event):
	view_port_main.input(event)
	view_port_minimap.input(event)
	print("k")
