extends Control

onready var module_notes_container = $ModuleNotesPanel/VBoxContainer/ModuleNotes
onready var image_container = $ImagePanel/VBoxContainer/ScrollContainer/ImageContainer
onready var legend = $ImagePanel/VBoxContainer/Legend
onready var value_zero_decoded_label = $ImagePanel/VBoxContainer/Legend/LegendItemZero/value_zero_decoded
onready var value_127_decoded_label = $ImagePanel/VBoxContainer/Legend/LegendItem127/value_127_decoded
onready var value_255_decoded_label = $ImagePanel/VBoxContainer/Legend/LegendItem255/value_255_decoded
export var text_scene : PackedScene
export var channel_scene : PackedScene
onready var image_scale_bar = $ModuleNotesPanel/VBoxContainer/ImageScaleBar
var n_cols : int setget set_n_cols
export var n_cols_inspector : int = 3
var module_notes_panel_visibility = true setget set_module_notes_panel_visibility
var network_module_resource : NetworkModuleResource setget set_network_module_resource
var legend_visible = true setget set_legend_visible
# feature_visualization_mode determines if feature visualizations are shown. 
# Otherwise activations are shown
export var feature_visualization_mode : bool = false setget set_feature_visualization_mode

# details_layout defines if module specific details are shown in the channel view.
# In detail mode each channel is shown in its own row. 
# When detail mode is inactive, the channels are shown in a grid (default)
export var details_layout = false setget set_details_layout

signal request_image_data(network_module_resource, feature_visualization_mode)


func _ready():
	var _error
	_error = connect("request_image_data", DLManager, "on_request_image_data")
	set_n_cols(n_cols_inspector)
	
	
func set_details_layout(mode : bool):
	details_layout = mode
	if details_layout:
		set_n_cols(1)
		$ImagePanel/VBoxContainer/ScrollContainer.scroll_vertical_enabled = false
		set_legend_visible(false)
	else:
		_on_image_scale_changed() # Set image columns
		$ImagePanel/VBoxContainer/ScrollContainer.scroll_vertical_enabled = true
		set_legend_visible(true)
	update_details_layout()
		

# Called by NetworkGroupSelector via group method. 
# By default, this node is not in this group.
func network_module_selected_by_detail_screen(network_module):
	var new_network_module_resource = network_module.network_module_resource
	if new_network_module_resource != network_module_resource:
		set_network_module_resource(new_network_module_resource)
	

func set_network_module_resource(new_network_module_resource):
	network_module_resource = new_network_module_resource
	update_text()
	emit_signal("request_image_data", network_module_resource, feature_visualization_mode)
	
	
# Called by DLManager via group
func receive_classification_results(_results):
	# When a new forward pass was performed, we need to update the image data.
	if not feature_visualization_mode and network_module_resource != null:
		emit_signal("request_image_data", network_module_resource, feature_visualization_mode)
		
		
# Called by DLManager via group
func set_fv_image_resource(_image_resource):
	if feature_visualization_mode:
		emit_signal("request_image_data", network_module_resource, feature_visualization_mode)
	
	
func update_text():
	for child in module_notes_container.get_children():
		child.queue_free()
	
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


# Called by DLManager when image data is received.
func receive_image_data(image_resource_data):
	if image_resource_data[0].module_id != network_module_resource.module_id:
		return
	var mode = ImageResource.MODE[image_resource_data[0]["mode"]]
	if mode == ImageResource.MODE.ACTIVATION and feature_visualization_mode == false or \
		mode == ImageResource.MODE.FEATURE_VISUALIZATION and feature_visualization_mode == true:
		add_to_group("on_pool_task_completed")
		THREADPOOL.submit_task(self, "process_module_image_resources", image_resource_data, "process_module_image_resources_" + String(network_module_resource.module_id))
		#var results = process_module_image_resources(image_resource_data)
		#update_image_data(results)


func process_module_image_resources(image_resource_data):
	var image_resources = []
	for item in image_resource_data:
		image_resources.append(ImageProcessing.dict_to_image_resource(item))
	# Also process other information: weights and rewiring
	return image_resources


# Called by THREADPOOL via Group
func on_pool_task_completed(task):
	if task.tag == "process_module_image_resources_" + String(network_module_resource.module_id):
		# Remove from THREADPOOL group. We assume that we use THREADPOOL sparsely and that we have no two running tasks at the same time.
		remove_from_group("on_pool_task_completed")
		call_deferred("update_image_data", task.result)
		

func update_image_data(image_resources=[]):
	set_loading(false)
	
	if not image_resources:
		# If no image_resources are given we reuse the data that is already present.
		for child in image_container.get_children():
			image_resources.append(child.image_resource)
			
	# Remove old channels
	for child in image_container.get_children():
		child.queue_free()
		
#	# Reset the vseparation of the image_container to default value
#	image_container.set("custom_constants/vseparation", 10)
#	# Set vseparation of the images.
#	var out_channels = network_module_resource.size[1]
#	var in_channels = out_channels
#	if network_module_resource.precursor_module_resources.size() > 0:
#		in_channels = network_module_resource.precursor_module_resources[0].size[1]
#	elif network_module_resource.input_channels > 0:
#		in_channels = network_module_resource.input_channels
#	if in_channels > out_channels:
#		var label_width = 0
#		if network_module_resource.channel_labels:
#			label_width = 40
#		image_container.set("custom_constants/vseparation", 10 + (256 + label_width) * (in_channels / out_channels - 1))
	
	var new_instance : Node
	var image_resource : ImageResource
	var _error
	for i in range(image_resources.size()):
		new_instance = channel_scene.instance()
		image_resource = image_resources[i]
		image_container.add_child(new_instance)
		new_instance.set_image_resource(image_resource)
		if details_layout:
			new_instance.set_network_module_resource(network_module_resource)
		# Connect signal such that we get notified when a weight is changed.
		_error = new_instance.connect("weight_changed", self, "on_weight_changed")
		
	# Update the legend
	if image_resource.value_zero_decoded != -1.0:
		var format_string = "%.*f"
		value_zero_decoded_label.text = format_string % [2, image_resource.value_zero_decoded]
		value_127_decoded_label.text = format_string % [2, (image_resource.value_zero_decoded + image_resource.value_255_decoded) / 2]
		value_255_decoded_label.text = format_string % [2, image_resource.value_255_decoded]
		set_legend_visible(legend_visible)
		
	else:
		legend.visible = false

	# Reset the scale bar
	image_scale_bar.value = 1.0


func set_n_cols(n_cols_):
	if not is_inside_tree() or not "columns" in image_container:
		pass
	n_cols = n_cols_
	image_container.columns = n_cols


func _on_image_scale_changed():
	if image_container.get_child_count() == 0:
		set_n_cols(n_cols_inspector)
		return
	
	# Change scale of image tiles.
	var scale_new = image_scale_bar.value
	var image_size = 256 * scale_new
	for child in image_container.get_children():
		child.set_size_of_children(image_size)
	
	# Change number of rows.
	if not details_layout:
		var width_of_image_grid = 256 * n_cols_inspector
		var h_separation = image_container.get("custom_constants/hseparation")
		var columns = int(width_of_image_grid / (image_size + h_separation))
		image_container.columns = columns
	
	
func set_module_notes_panel_visibility(mode : bool):
	module_notes_panel_visibility = mode
	$ModuleNotesPanel.visible = mode


func update_details_layout():
	for child in image_container.get_children():
		if details_layout:
			child.set_network_module_resource(network_module_resource)
		else:
			child.set_network_module_resource(null)


func set_feature_visualization_mode(feature_visualization_mode_):
	feature_visualization_mode = feature_visualization_mode_
	if network_module_resource != null:
		# If we already have a network module resource we update the images.
		emit_signal("request_image_data", network_module_resource, feature_visualization_mode)
	
	
func set_legend_visible(mode : bool):
	legend_visible = mode
	$ImagePanel/VBoxContainer/Legend.visible = mode

# Called by signal when a weight is changed by a channel node.
func on_weight_changed(weight, channel_ind, group_ind):
	#var group_size = network_module_resource.weights[0].size()
	network_module_resource.weights[channel_ind][group_ind][0][0] = weight
	get_tree().call_group("on_weight_changed", "weight_changed", network_module_resource)
	

# Called by NetworkModuleActionSelector
func zero_weights():
	if "weights" in network_module_resource:
		for channel_weights in network_module_resource.weights:
			for group_weight in channel_weights:
				group_weight[0][0] = 0.0
	get_tree().call_group("on_weight_changed", "weight_changed", network_module_resource)
	update_details_layout()
	
	
# Called by NetworkModuleActionSelector
func identity_weights():
	if "weights" in network_module_resource:
		var group = 0
		var group_size = network_module_resource.weights[0].size()
		for channel_weights in network_module_resource.weights:
			for i in range(group_size):
				if group == i:
					channel_weights[i][0][0] = 1.0
				else:
					channel_weights[i][0][0] = 0.0
			group = (group + 1) % group_size
	get_tree().call_group("on_weight_changed", "weight_changed", network_module_resource)
	update_details_layout()


func loading_fv_data(module_id):
	if feature_visualization_mode and network_module_resource != null and module_id == network_module_resource.module_id:
		for child in image_container.get_children():
			child.queue_free()
		set_loading(true)
		
		
func set_loading(mode : bool):
	$LoadingScreen.visible = mode
	$ImagePanel.visible = not mode
