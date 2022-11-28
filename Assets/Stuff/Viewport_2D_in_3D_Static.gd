tool
extends Spatial

export var screen_size = Vector2(3.0, 2.0) setget set_screen_size
export var viewport_size = Vector2(300.0, 200.0) setget set_viewport_size
export var transparent = true setget set_transparent
export var scene : PackedScene = null setget set_scene, get_scene

var is_ready : bool = false
var scene_node : Node


func set_screen_size(new_size: Vector2):
	screen_size = new_size
	if is_ready:
		$Screen.mesh.size = screen_size


func set_viewport_size(new_size: Vector2):
	viewport_size = new_size
	if is_ready:
		$Viewport.size = new_size
		var material : SpatialMaterial = $Screen.get_surface_material(0)
		# When the viewport is just created, it will have no valid texture just yet.
		# Wait a frame.
		yield(VisualServer, "frame_post_draw")
		material.albedo_texture = $Viewport.get_texture()


func set_transparent(new_transparent: bool):
	transparent = new_transparent
	if is_ready:
		var material : SpatialMaterial = $Screen.get_surface_material(0)
		material.flags_transparent = transparent
		$Viewport.transparent_bg = transparent


func set_scene(new_scene: PackedScene):
	scene = new_scene
	if is_ready:
		# out with the old
		if scene_node:
			$Viewport.remove_child(scene_node)
			scene_node.queue_free()

		# in with the new
		if scene:
			scene_node = scene.instance()
			$Viewport.add_child(scene_node)

func get_scene():
	return scene

func get_scene_instance():
	return scene_node
	
func connect_scene_signal(which, on, callback):
	if scene_node:
		var _error
		_error = scene_node.connect(which, on, callback)

# Called when the node enters the scene tree for the first time.
func _ready():
	# apply properties
	is_ready = true
	set_screen_size(screen_size)
	set_viewport_size(viewport_size)
	set_scene(scene)
	set_transparent(transparent)
