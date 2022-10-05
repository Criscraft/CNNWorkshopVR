extends Spatial

export var spawn_position_node_paths : NodePath
var pickable_image_scene

signal get_dataset_images(ids)

# Called when the node enters the scene tree for the first time.
func _ready():
	pickable_image_scene = preload("res://Assets/DL/PickableImage.tscn")


func _on_DLManager_on_connected():
	var instance
	var spawn_position_node 
	#var image
	#var resource
	#var dataset = $DatasetManager
	
	var spawn_position_nodes = get_node(spawn_position_node_paths).get_children()
	
	for i in range(spawn_position_nodes.size()):
		spawn_position_node = spawn_position_nodes[i]
		# Create new instance, move and add to tree
		instance = pickable_image_scene.instance()
		instance.global_transform = spawn_position_node.global_transform
		add_child(instance)
		
		# Get image resource
		#resource = dataset.get_image(i)
	
		# Request image data
	emit_signal("get_dataset_images", range(spawn_position_nodes.size()))
	
func _on_DLManager_receive_dataset_images(image_resources):
	var i = 0
	for child in get_children():
		child.get_node("ImageLogic").image_resource = image_resources[i]
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






