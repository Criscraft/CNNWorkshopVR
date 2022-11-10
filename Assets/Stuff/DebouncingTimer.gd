extends Timer


func _on_trigger(_value=null):
	if is_stopped():
		start()

