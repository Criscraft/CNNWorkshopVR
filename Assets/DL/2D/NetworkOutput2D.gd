extends Control

onready var results = $VBoxContainer/Results
var result_label_scene : PackedScene = preload("res://Assets/Stuff/TextLine.tscn")


func _ready():
	add_result_line("Welcome!")
	add_result_line("1. Take image from the left.")
	add_result_line("2. Put image on the table.")
	add_result_line("3. See network output.")
	add_result_line("4. Enjoy yourself!")


func add_result_line(result_line : String):
	var new_line = result_label_scene.instance()
	new_line.text = result_line
	results.add_child(new_line)
	
	
func add_result_node(node : Node):
	results.add_child(node)
	

func clear_results():
	for child in results.get_children():
		child.queue_free()
