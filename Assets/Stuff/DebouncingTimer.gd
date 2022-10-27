extends Timer


func _on_trigger(_value):
	if is_stopped():
		start()
		
		
