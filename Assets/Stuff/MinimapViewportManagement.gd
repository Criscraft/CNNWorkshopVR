tool
extends Node2D

export var viewport_minimap_path : NodePath
onready var viewport_minimap = get_node(viewport_minimap_path)
export var camera_minimap_path : NodePath
onready var camera_minimap = get_node(camera_minimap_path)

export var viewport_main_path : NodePath
onready var viewport_main = get_node(viewport_main_path)
export var camera_main_path : NodePath
onready var camera_main = get_node(camera_main_path)

# Called when the node enters the scene tree for the first time.
func _ready():
	viewport_minimap.world_2d = viewport_main.world_2d
	camera_minimap.custom_viewport = viewport_minimap
	camera_minimap.current = true
	camera_main.current = true
	
