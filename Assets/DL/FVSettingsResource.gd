extends Resource
class_name FVSettingsResource

enum MODE {AVERAGE, CENTERPIXEL, PERCENTILE}
enum POOLMODE {avgpool, maxpool, interpolate_antialias, interpolate, subsample, identity, identity_smooth}


export var mode : int = MODE.PERCENTILE
export var epochs : int = 100
export var epochs_without_robustness_transforms : int = 0
export var lr : float = 20.0
export var degrees : int = 0
export var blur_sigma : float = 0.5
export var roll : int = 0
export var fraction_to_maximize : float = 0.25
export var pool_mode : int = POOLMODE.avgpool
export var filter_mode : bool = false

func get_dict():
	var out = {
		"mode" : MODE.keys()[mode],
		"epochs" : epochs,
		"epochs_without_robustness_transforms" : epochs_without_robustness_transforms,
		"lr" : lr,
		"degrees" : degrees,
		"blur_sigma" : blur_sigma,
		"roll" : roll,
		"fraction_to_maximize" : fraction_to_maximize,
		"pool_mode" : POOLMODE.keys()[pool_mode],
		"filter_mode" : filter_mode,
	}
	return out
	
