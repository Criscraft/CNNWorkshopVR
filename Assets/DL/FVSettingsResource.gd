extends Resource
class_name FVSettingsResource

enum MODE {AVERAGE, CENTERPIXEL, PERCENTILE}

export var mode : int = MODE.CENTERPIXEL
export var epochs : int = 200
export var epochs_without_robustness_transforms : int = 10
export var lr : float = 20.0
export var degrees : int = 10
export var blur_sigma : float = 0.5
export var roll : int = 4
export var fraction_to_maximize : float = 0.25

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
	}
	return out
	
