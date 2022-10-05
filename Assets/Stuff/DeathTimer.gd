extends Timer

func _on_DeathTimer_timeout():
	get_parent().queue_free()
