extends Spatial

export var floor_height : float = 2.5
export var n_floors : int = 2
# The children are interpreted as floors.
# The first child is the lowest floor.
export var current_floor = 0 setget set_current_floor
onready var platform = $Platform


func increment_floor(_at=null):
	set_current_floor(current_floor + 1)
		
		
func decrement_floor(_at=null):
	set_current_floor(current_floor - 1)
		
		
func set_current_floor(new_floor):
	if new_floor >= 0 and new_floor < n_floors:
		current_floor = new_floor
		platform.translation.y = current_floor * floor_height
