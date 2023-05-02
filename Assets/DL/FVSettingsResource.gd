extends Resource
class_name FVSettingsResource

enum MODE {AVERAGE, CENTERPIXEL, PERCENTILE}
enum POOLMODE {avgpool, maxpool, interpolate_antialias, interpolate, subsample, identity, identity_smooth, lppool, undefined}


export var mode : int = MODE.PERCENTILE
export var epochs : int = 100
export var lr : float = 20.0
export var degrees : int = 0
export var blur_sigma : float = 0.5
export var roll : int = 0
export var fraction_to_maximize : float = 0.25
export var pool_mode : int = POOLMODE.undefined
export var mimic_poolstage_filter_size : bool = false
export var slope_leaky_relu_scheduling : bool = true
export var final_slope_leaky_relu : float = 0.01

func get_dict():
	var out = {
		"mode" : MODE.keys()[mode],
		"epochs" : epochs,
		"lr" : lr,
		"degrees" : degrees,
		"blur_sigma" : blur_sigma,
		"roll" : roll,
		"fraction_to_maximize" : fraction_to_maximize,
		"pool_mode" : POOLMODE.keys()[pool_mode],
		"mimic_poolstage_filter_size" : mimic_poolstage_filter_size,
		"slope_leaky_relu_scheduling" : slope_leaky_relu_scheduling,
		"final_slope_leaky_relu" : final_slope_leaky_relu,
	}
	return out
	
