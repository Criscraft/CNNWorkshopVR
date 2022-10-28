tool
extends Spatial

export var text = "Insert text."

# Called when the node enters the scene tree for the first time.
func _ready():
	var scene_instance = $Viewport2Din3DStatic.get_scene_instance()
	scene_instance.text = text
	scene_instance.align = 1
