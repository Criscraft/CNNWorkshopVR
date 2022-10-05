extends Spatial

# Check if OpenXR can be initiallized. If so, load the VR player. If not, load the desktop player.

var interface : ARVRInterface = ARVRServer.find_interface("OpenXR")

func _ready():
	if interface and interface.initialize():
		print("OpenXR Interface available")
		interface.uninitialize()
		var resource = load("res://Assets/VRPlayer/first_person_controller_vr.tscn")
		var scene_instance = resource.instance()
		$Player.add_child(scene_instance)
	else:
		print("OpenXR Interface initialization failed. Load desktop Player")
		var resource = load("res://Assets/DesktopPlayer/DesktopPlayer.tscn")
		var scene_instance = resource.instance()
		$Player.add_child(scene_instance)
		
