extends Resource
class_name NetworkModuleResource

enum TYPE {
	INPUT,
	OUTPUT,
	GROUPCONV, # connections inside group only, editable weights
	SIMPLENODE, # show activations
	REWIRE, # input - output rewiring
	COPY,
	ADD, # merge two branches
	POOL,
	RELU,
	CONV,
	HFCONV, # shows kernel and output, when f>1 one input is linked to several kernels
	NORM,
}

# Guaranteed to be filled with values
export var module_id : int = -1
export var tracker_module_type : int = -1
export var group_id : int = -1
export var label : String = ""
export var precursors : Array = []
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
	
