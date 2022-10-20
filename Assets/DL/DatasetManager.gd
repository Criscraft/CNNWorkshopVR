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
	
	var _error
	# When the DLManager connects we request the dataset images.
	_error = DLManager.connect("on_connected", self, "_on_DLManager_connected")
	# When we request the dataset images, we want to send a signal to the DLManager
	_error = connect("request_dataset_images", DLManager, "on_request_dataset_images")


func _on_DLManager_connected():
	emit_signal("request_dataset_images", $SpawnPositionNodes.get_child_count())
	
	
func preprocess_dataset_images(image_resource_data):
	var image_resources = []
	for item in image_resource_data:
		image_resources.append(ImageProcessing.dict_to_image_resource(item))
	return image_resources
	

# Called by DLManager via Group
func receive_dataset_images(image_resource_data):
	add_to_group("on_pool_task_completed")
	THREADPOOL.submit_task(self, "preprocess_dataset_images", image_resource_data, "preprocess_dataset_images")


# Called by THREADPOOL via Group
func on_pool_task_completed(task):
	if task.tag == "preprocess_dataset_images":
		# Remove from THREADPOOL group. We assume that we use THREADPOOL sparsely and that we have no two running tasks at the same time.
		remove_from_group("on_pool_task_completed")
		call_deferred("on_finished_preprocess_dataset_images", task.result)


func on_finished_preprocess_dataset_images(image_resources):
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
