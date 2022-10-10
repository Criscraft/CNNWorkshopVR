extends Node2D


export var network_group_scene : PackedScene


func _ready():
	var new_instance = network_group_scene.instance()
	add_child(new_instance)


func receive_architecture(architecture_dict):
	print("received_architecture")
