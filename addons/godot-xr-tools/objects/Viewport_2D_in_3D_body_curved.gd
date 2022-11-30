extends "res://addons/godot-xr-tools/objects/Viewport_2D_in_3D_body.gd"

export var screen_angle_range : int = 180

func _post_ready():
	vp = get_node("../../Viewport")

# Convert intersection point to screen coordinate
func global_to_viewport(p_at):
	var t = $CollisionShape.global_transform
	var at = t.xform_inv(p_at)
	
	var radius = at.length()
	var angle = asin(at.x / radius)
	var screen_angle_range_rad = deg2rad(screen_angle_range)
	at.x = angle / screen_angle_range_rad # ranges from -0.5 to 0.5
	at.y = at.y / screen_size.y # ranges from -0.5 to 0.5
	
	# Convert to screen space
	at.x = (at.x + 0.5) * viewport_size.x
	at.y = (0.5 - at.y) * viewport_size.y
	
	return Vector2(at.x, at.y)
