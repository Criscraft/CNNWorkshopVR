extends StaticBody

var pickable_image_scene

signal request_dataset_images(n)

# Called when the node enters the scene tree for the first time.
func _ready():
	pickable_image_scene = preload("res://Assets/DL/PickableImage.tscn")
	# Create snap trays for the images and the images.
	var children = $SpawnPositionNodes.get_children()
	var snap_tray_scene = preload("res://Assets/Stuff/MySnapTray.tscn")
	var instance = null
	for child in children:
		# Create snap tray
		instance = snap_tray_scene.instance()
		child.add_child(instance)
	#yield(get_tree(), "idle_frame") # Wait to just before _process is called. Otherwise we cannot add the following pickable images to the Staging node


func _on_DLManager_on_connected():
	emit_signal("request_dataset_images", $SpawnPositionNodes.get_child_count())
	
	
func _on_DLManager_receive_dataset_images(image_resources):
	var snap_tray
	var instance
	var i = 0
	var spawned_items_node = get_tree().get_root().get_node("Staging/SpawnedItems")
	for spawn_position_node in $SpawnPositionNodes.get_children():
		# Delete old PickableImage 
		snap_tray = spawn_position_node.get_child(0).get_node("Snap_Zone")
		snap_tray.destroy_held_item()
		# Create new instance, move and add to tree
		instance = pickable_image_scene.instance()
		instance.global_transform = spawn_position_node.global_transform
		spawned_items_node.add_child(instance)
		# After the pickable image is added to the scene tree, we can set its resource image.
		instance.get_node("ImageLogic").image_resource = image_resources[i]
		i += 1
		

"""
func mytest():
	var random = RandomNumberGenerator.new()
	random.randomize()
	for child in children:
		
		# Create new instance, move and add to tree
		instance = pickable_image_scene.instance()
		instance.global_transform = child.global_transform
		get_node("/root/Staging").add_child(instance)
		
		# Create image
		image = Image.new()
		image.create(224, 224, false, Image.FORMAT_RGB8)
		image.lock()
		for i in range(image.get_height()):
			for j in range(image.get_width()):
				image.set_pixel(i, j, Color(random.randf(), random.randf(), random.randf()))
		image.unlock()
		# Create image resource
		resource = DLImageResource.new() 
		resource.image = image
		resource.mode = DLImageResource.MODE.DATASET
		
		# Apply image data
		instance.get_node("ImageLogic").image_resource = resource
"""
