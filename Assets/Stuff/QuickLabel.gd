tool
extends Spatial

export var text = "Insert text." setget set_text

# scene_instance is a TextLine instance.
#onready var scene_instance = $Viewport2Din3DStatic.get_scene_instance()


func _ready():
	set_text(text)

func set_text(text_):
	text = text_
	var scene_instance = get_node("Viewport2Din3DStatic").get_scene_instance()
	if scene_instance != null:
		scene_instance.text = text
		scene_instance.align = 1
	
