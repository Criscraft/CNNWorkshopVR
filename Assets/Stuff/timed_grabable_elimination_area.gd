extends Node

export var death_timer_path : String
var death_timer_scene : Resource

func _ready():
	death_timer_scene = load(death_timer_path)


func _on_TimedGrabableEliminationArea_body_entered(body):
	var instance = death_timer_scene.instance()
	body.add_child(instance)
