extends StaticBody

# Called when the node enters the scene tree for the first time.
func _ready():
	# Create snap trays for the images and the images.
	var children = $SpawnPositionNodes.get_children()
	var snap_tray_scene = preload("res://Assets/Stuff/MySnapTray.tscn")
	var instance = null
	for child in children:
		# Create snap tray
		instance = snap_tray_scene.instance()
		child.add_child(instance)
	#yield(get_tree(), "idle_frame") # Wait to just before _process is called. Otherwise we cannot add the following pickable images to the Staging node
	
