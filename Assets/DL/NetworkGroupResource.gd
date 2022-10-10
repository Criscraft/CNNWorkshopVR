extends Resource
class_name NetworkGroupResource

enum TYPE {INPUT, OUTPUT, GROUPCONV, HFModule, REWIRE, COPY, ADD, POOL, SUM, SIMPLEGROUP}

export var id : int = -1
export var tracker_module_group_type : int = -1
export var precursors : Array
export var label : String = ""

func get_dict():
	var out = {
		"id" : id,
		"tracker_module_group_type" : TYPE.keys()[tracker_module_group_type],
		"precursors" : precursors,
		"label" : label,
	}
	return out
	
