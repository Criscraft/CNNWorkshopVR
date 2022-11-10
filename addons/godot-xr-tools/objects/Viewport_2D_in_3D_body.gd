extends XRToolsInteractableBody

export var screen_size = Vector2(3.0, 2.0)
export var viewport_size = Vector2(100.0, 100.0)
var last_cursor_position = Vector2.ZERO
var last_cursor_global_position = Vector2.ZERO

var vp = null

# Called when the node enters the scene tree for the first time.
func _ready():
	call_deferred("_post_ready")
	
func _post_ready():
	vp = get_node("../Viewport")

# Convert intersection point to screen coordinate
func global_to_viewport(p_at):
	var t = $CollisionShape.global_transform
	var at = t.xform_inv(p_at)
	
	# Convert to screen space
	at.x = ((at.x / screen_size.x) + 0.5) * viewport_size.x
	at.y = (0.5 - (at.y / screen_size.y)) * viewport_size.y
	
	return Vector2(at.x, at.y)

func _on_pointer_moved(from, to):
	if not vp:
		return
	
	var local_from = global_to_viewport(from)
	var local_to = global_to_viewport(to)
	last_cursor_position = local_to
	last_cursor_global_position = local_to
	
	# Let's mimic a mouse
	var event = InputEventMouseMotion.new()
	event.set_position(local_to)
	event.set_global_position(local_to)
	event.set_relative(local_to - local_from) # should this be scaled/warped?
	event.set_button_mask(BUTTON_MASK_LEFT)
	vp.input(event)

func _on_pointer_pressed(at):
	if not vp:
		return
		
	var local_at = global_to_viewport(at)
	
	# Let's mimic a mouse
	#mouse_mask = 1
	var event = InputEventMouseButton.new()
	event.set_button_index(BUTTON_LEFT)
	event.set_pressed(true)
	event.set_position(local_at)
	event.set_global_position(local_at)
	event.set_button_mask(BUTTON_MASK_LEFT)
	vp.input(event)
	

func _on_pointer_released(at):
	if not vp:
		return
		
	var local_at = global_to_viewport(at)
	
	# Let's mimic a mouse
	#mouse_mask = 0
	var event = InputEventMouseButton.new()
	event.set_button_index(BUTTON_LEFT)
	event.set_pressed(false)
	event.set_position(local_at)
	event.set_global_position(local_at)
	event.set_button_mask(BUTTON_MASK_LEFT)
	vp.input(event)

