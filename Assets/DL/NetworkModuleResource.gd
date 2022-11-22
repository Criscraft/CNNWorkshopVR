extends Resource
class_name NetworkModuleResource

# Guaranteed to be filled with values
export var module_id : int = -1
export var group_id : int = -1
export var label : String = ""
export var precursors : Array = []
export var precursor_module_resources : Array = []
export var has_data : bool = false

# Optional properties
export var channel_labels : Array = []
export var size : Array = []
export var weights : Array = []
export var weights_min : float = -1
export var weights_max : float = -1
export var permutation : Array = []
export var kernels : Array = []
export var padding : int = -1


func get_dict():
	# Export only data that can be used to identify the module
	var out = {
		"module_id" : module_id,
		"group_id" : group_id,
	}
	return out
	
