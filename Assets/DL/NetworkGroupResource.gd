extends Resource
class_name NetworkGroupResource

enum TYPE {
	INPUT, 
	OUTPUT, 
	GROUPCONV, 
	HFModule, 
	REWIRE, 
	COPY, 
	ADD, 
	POOL, 
	SUM, 
	SIMPLEGROUP
}

export var group_id : int = -1
export var tracker_module_group_type : int = -1
export var precursors : Array = []
export var label : String = ""
	

func get_dict():
	# Export only data that can be used to identify the network group
	var out = {
		"group_id" : group_id,
		"tracker_module_group_type" : TYPE.keys()[tracker_module_group_type],
	}
	return out
	
