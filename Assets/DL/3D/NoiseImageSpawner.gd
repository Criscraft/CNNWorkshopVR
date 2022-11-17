extends Spatial

export var pickable_image_scene : PackedScene
onready var snap_zone = $MySnapTray/Snap_Zone


func _ready():
	var _error
	_error = DLManager.connect("on_receive_noise_image", self, "spawn_image")


func spawn_image(image_dict):
	var image_resource = ImageProcessing.dict_to_image_resource(image_dict)
	# Empty the snap tray
	snap_zone.destroy_held_item()
	# Create new portable image instance
	var instance = pickable_image_scene.instance()
	instance.global_transform = snap_zone.global_transform
	var spawned_items_node = get_tree().get_root().get_node("Staging/SpawnedItems")
	spawned_items_node.add_child(instance)
	# After the pickable image is added to the scene tree, we can set its resource image.
	instance.get_node("ImageLogic").image_resource = image_resource
