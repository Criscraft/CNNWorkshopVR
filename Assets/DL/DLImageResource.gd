extends Resource
class_name DLImageResource

enum MODE {DATASET, ACTIVATION, FEATURE_VISUALIZATION, NOISE}

export var id : int = -1
export var module_id : int = -1
export var channel_id : int = -1
export var mode : int = -1
export var label : String = ""
export var image : Image = null

func get_dict(get_image=false):
	# Export only data that can be used to identify the image
	var out = {
		"id" : id,
		"module_id" : module_id,
		"channel_id" : channel_id,
		"mode" : MODE.keys()[mode],
	}
	if get_image:
		pass
	return out
	
