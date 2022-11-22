extends Resource
class_name NetworkGroupResource

export var group_id : int = -1
export var precursors : Array = []
export var label : String = ""
export var precursor_group_resources : Array = []
	

func get_dict():
	# Export only data that can be used to identify the network group
	var out = {
		"group_id" : group_id,
	}
	return out
	
