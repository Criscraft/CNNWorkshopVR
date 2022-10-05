extends Resource
class_name DLImageResource

enum MODE {DATASET, ACTIVATION, FEATURE_VISUALIZATION, NOISE}

export var id : int
export var module_id : int
export var channel_id : int
export var mode : int
export var label : String
export var image : Image
