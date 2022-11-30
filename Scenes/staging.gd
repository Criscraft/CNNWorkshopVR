extends Spatial

# Check if OpenXR can be initiallized. If so, load the VR player. If not, load the desktop player.

var interface : ARVRInterface = ARVRServer.find_interface("OpenXR")
export var debug : bool = false
export var vr_player_scene : PackedScene = preload("res://Assets/VRPlayer/first_person_controller_vr.tscn")
export var desktop_player_scene : PackedScene = preload("res://Assets/DesktopPlayer/DesktopPlayer.tscn")
export var debug_desktop_player_scene  : PackedScene = preload("res://Assets/Debug/DebugPlayer.tscn")

func _ready():
	if interface and interface.initialize():
		print("OpenXR Interface available")
		interface.uninitialize()
		var scene_instance = vr_player_scene.instance()
		$Player.add_child(scene_instance)
	else:
		print("OpenXR Interface initialization failed. Load desktop Player")
		var scene_instance
		if debug:
			scene_instance = debug_desktop_player_scene.instance()
		else:
			scene_instance = desktop_player_scene.instance()
		$Player.add_child(scene_instance)
		
