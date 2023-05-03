extends CollisionObject

var button_albedo : Color

signal pointer_pressed(at)

func _ready():
	print($MeshInstance)
	button_albedo = $MeshInstance.get_surface_material(0).get("albedo_color")

func _on_Button_pointer_pressed(at):
	# Deactivate pointable button
	$CollisionShape.disabled = true
	
	# Color animation
	var tween = get_tree().create_tween()
	tween.tween_property($MeshInstance.get_surface_material(0), "albedo_color", Color.red, 0.2)
	tween.tween_property(null, "", null, 1.0)
	tween.tween_property($MeshInstance.get_surface_material(0), "albedo_color", button_albedo, 0.2)
	tween.tween_callback(self, "activate_button")
	
func activate_button():
	# Reactivate pointable button
	$CollisionShape.disabled = false
	#$MeshInstance.get_surface_material(0).set("albedo_color", Color(255, 0, 0))
