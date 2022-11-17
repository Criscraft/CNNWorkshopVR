extends Node

export var death_timer_scene : PackedScene = preload("res://Assets/Stuff/DeathTimer.tscn")


func _on_TimedGrabableEliminationArea_body_entered(body):
	var instance = death_timer_scene.instance()
	body.add_child(instance)
