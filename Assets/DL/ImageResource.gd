extends Resource
class_name ImageResource

enum MODE {DATASET, ACTIVATION, FEATURE_VISUALIZATION, NOISE}

export var id : int = -1
export var module_id : int = -1
export var channel_id : int = -1
export var mode : int = -1
export var label : String = "" # is used by pickable image logic, not image logic 2d
export var image : Image = null
export var value_zero_decoded : float = -1.0
export var value_255_decoded : float = -1.0

func get_dict(get_image=false):
	# Export only data that can be used to identify the image
	var out = {
		"id" : id,
		"module_id" : module_id,
		"channel_id" : channel_id,
		"mode" : MODE.keys()[mode],
	}
	if get_image and image != null:
		var png = image.save_png_to_buffer()
		var image_encoded = Marshalls.raw_to_base64(png)
		out["data"] = image_encoded
	return out
	
