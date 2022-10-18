tool
extends Node2D

export var viewport_minimap_path : NodePath
export var camera_minimap_path : NodePath
export var viewport_main_path : NodePath

# Called when the node enters the scene tree for the first time.
func _ready():
	var viewport_minimap = get_node(viewport_minimap_path)
	var camera_minimap = get_node(camera_minimap_path)
	var viewport_main = get_node(viewport_main_path)
	viewport_minimap.world_2d = viewport_main.world_2d
	camera_minimap.custom_viewport = viewport_minimap
	camera_minimap.current = true
