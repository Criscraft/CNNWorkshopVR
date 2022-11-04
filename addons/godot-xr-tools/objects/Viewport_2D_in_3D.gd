tool
extends Spatial

signal pointer_entered
signal pointer_exited

export var enabled = true setget set_enabled
export var screen_size = Vector2(3.0, 2.0) setget set_screen_size
export var viewport_size = Vector2(300.0, 200.0) setget set_viewport_size
export var transparent = true setget set_transparent
export (PackedScene) var scene = null setget set_scene, get_scene

# Need to replace this with proper solution once support for layer selection has been added
export (int, LAYERS_3D_PHYSICS) var collision_layer = 15 setget set_collision_layer

var is_ready = false
var scene_node = null
var is_pointer_entered = false

func set_enabled(is_enabled: bool):
	enabled = is_enabled
	if is_ready:
		$StaticBody/CollisionShape.disabled = !enabled

func set_screen_size(new_size: Vector2):
	screen_size = new_size
	if is_ready:
		$Screen.mesh.size = screen_size
		$StaticBody.screen_size = screen_size
		$StaticBody/CollisionShape.shape.extents = Vector3(screen_size.x * 0.5, screen_size.y * 0.5, 0.01)

func set_viewport_size(new_size: Vector2):
	viewport_size = new_size
	if is_ready:
		$Viewport.size = new_size
		$StaticBody.viewport_size = new_size
		var material : SpatialMaterial = $Screen.get_surface_material(0)
		yield(VisualServer, "frame_post_draw")
		material.albedo_texture = $Viewport.get_texture()

func set_transparent(new_transparent: bool):
	transparent = new_transparent
	if is_ready:
		var material : SpatialMaterial = $Screen.get_surface_material(0)
		material.flags_transparent = transparent
		$Viewport.transparent_bg = transparent

func set_collision_layer(new_layer: int):
	collision_layer = new_layer
	if is_ready:
		$StaticBody.collision_layer = collision_layer

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
			if scene_node is Scene2DIn3D:
				connect("pointer_exited", scene_node, "_on_vp_pointer_exited")
				connect("pointer_entered", scene_node, "_on_vp_pointer_entered")

func get_scene():
	return scene

func get_scene_instance():
	return scene_node

func connect_scene_signal(which, on, callback):
	if scene_node:
		scene_node.connect(which, on, callback)

# Called when the node enters the scene tree for the first time.
func _ready():
	# apply properties
	is_ready = true
	set_enabled(enabled)
	set_screen_size(screen_size)
	set_viewport_size(viewport_size)
	set_scene(scene)
	set_collision_layer(collision_layer)
	set_transparent(transparent)
	set_process_input(true)

func _on_pointer_entered():
	is_pointer_entered = true
	emit_signal("pointer_entered")
	
func _on_pointer_exited():
	is_pointer_entered = false
	emit_signal("pointer_exited")
	
func _input(event):
	if not is_pointer_entered:
		return
		
	# Forward scroll_wheel action to the viewport.
	if event is InputEventMouseButton and \
	event.pressed and \
	(event.button_index == BUTTON_WHEEL_UP or \
	event.button_index == BUTTON_WHEEL_DOWN):
		$Viewport.input(event)
		
	# Block other mouse input because we want the mouse input to be emulated by the body.
	# Real mouse inputs should not pass.
	if event is InputEventMouseMotion or event is InputEventMouseButton:
		return
		
	# Forward other events.
	$Viewport.input(event)
