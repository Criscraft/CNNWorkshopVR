extends Node
class_name Scene2DIn3D

signal pointer_entered
signal pointer_exited

func _on_vp_pointer_entered():
	emit_signal("pointer_entered")
	
func _on_vp_pointer_exited():
	emit_signal("pointer_exited")
