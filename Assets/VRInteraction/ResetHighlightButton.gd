extends Spatial


func _on_pointer_pressed(_at=null):
	call_deferred("request_highlight_update")
	
	
func request_highlight_update():
	ArchitectureManager.channel_highlighting.set_highlights_zero_and_apply()
