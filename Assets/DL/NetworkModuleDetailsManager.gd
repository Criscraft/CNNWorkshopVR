extends Control

onready var module_notes_container = get_node("HSplitContainer/Panel/ModuleNotes")
onready var image_grid_container = get_node("HSplitContainer/Panel2/ScrollContainer/ImageGrid")
export var text_scene_path : String = "res://Assets/DL/ResultLabel.tscn"
onready var text_scene : PackedScene = load(text_scene_path)
export var image_tile_scene_path : String = "res://Assets/DL/ImageTile.tscn"
onready var image_tile_scene : PackedScene = load(image_tile_scene_path)
var n_cols : int setget set_n_cols
export var n_cols_inspector : int = 3

var new_network_resource_candidate : NetworkModuleResource
var network_module_resource : NetworkModuleResource

signal request_image_data(network_module_resource, mode)

func _ready():
	var _error
	_error = connect("request_image_data", DLManager, "on_request_image_data")
	set_n_cols(n_cols_inspector)

func network_module_selected_by_detail_screen(network_module):
	if "network_module_resource" in network_module and network_module.network_module_resource != network_module_resource:
		new_network_resource_candidate = network_module.network_module_resource
	

func _on_UpdateModuleSelection_timeout():
	if new_network_resource_candidate != network_module_resource:
		update_network_module_resource()


func update_network_module_resource():
	network_module_resource = new_network_resource_candidate
	update_text()
	emit_signal("request_image_data", network_module_resource, "activation")
	
	
func update_text():
	for child in module_notes_container.get_children():
		module_notes_container.remove_child(child)
	
	add_text(network_module_resource.label)
	
	if network_module_resource.size:
		add_text("Tensor size:")
		add_text(String(network_module_resource.size.slice(1,-1)))
		
	if network_module_resource.weights:
		add_text("Weights size:")
		add_text(String(network_module_resource.weights.size()))
		
	if network_module_resource.kernels:
		add_text("Number of kernels:")
		add_text(String(network_module_resource.kernels.size()))
		add_text("Kernel size:")
		add_text(
			String(network_module_resource.kernels[0].size()) +
			" x " +
			String(network_module_resource.kernels[0][0].size()) +
			" x " +
			String(network_module_resource.kernels[0][0][0].size())
		)
		
	if network_module_resource.padding > -1:
		add_text("Padding size:")
		add_text(String(network_module_resource.padding))
		
		
func add_text(text):
	var new_text = text_scene.instance()
	new_text.text = text
	module_notes_container.add_child(new_text)


func receive_image_data(image_resource_data):
	add_to_group("on_pool_task_completed")
	#THREADPOOL.submit_task(self, "process_module_image_resources", image_resource_data, "process_module_image_resources")
	var results = process_module_image_resources(image_resource_data)
	on_finished_process_module_image_resources(results)


func process_module_image_resources(image_resource_data):
	var image_resources = []
	for item in image_resource_data:
		image_resources.append(ImageProcessing.dict_to_image_resource(item))
	var image_tiles = []
	for image_resource in image_resources:
		image_tiles.append(image_tile_scene.instance())
	return {"image_resources" : image_resources, "image_tiles" : image_tiles}


# Called by THREADPOOL via Group
func on_pool_task_completed(task):
	if task.tag == "process_module_image_resources":
		# Remove from THREADPOOL group. We assume that we use THREADPOOL sparsely and that we have no two running tasks at the same time.
		remove_from_group("on_pool_task_completed")
		call_deferred("on_finished_process_module_image_resources", task.result)
		

func on_finished_process_module_image_resources(results_dict):
	for image_tile in image_grid_container.get_children():
		image_grid_container.remove_child(image_tile)
	var image_tiles = results_dict["image_tiles"]
	var image_resources = results_dict["image_resources"]
	var image_tile
	var image_resource
	for i in range(image_resources.size()):
		image_tile = image_tiles[i]
		image_resource = image_resources[i]
		image_grid_container.add_child(image_tile)
		image_tile.image_resource = image_resource


func set_n_cols(n_cols_):
	if not is_inside_tree():
		pass
	n_cols = n_cols_
	$HSplitContainer/Panel2/ScrollContainer/ImageGrid.columns = n_cols
