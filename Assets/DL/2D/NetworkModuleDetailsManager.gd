extends Control

onready var module_notes_container = $ModuleNotesPanel/VBoxContainer/ModuleNotes
onready var channel_container = $ImagePanel/VBoxContainer/ScrollContainer/HBoxContainer/ChannelContainer
onready var legend = $ImagePanel/VBoxContainer/Legend
onready var value_zero_decoded_label = $ImagePanel/VBoxContainer/Legend/LegendItemZero/value_zero_decoded
onready var value_127_decoded_label = $ImagePanel/VBoxContainer/Legend/LegendItem127/value_127_decoded
onready var value_255_decoded_label = $ImagePanel/VBoxContainer/Legend/LegendItem255/value_255_decoded
onready var padding_rect = $ImagePanel/VBoxContainer/ScrollContainer/HBoxContainer/Padding
var text_scene : PackedScene = preload("res://Assets/Stuff/TextLine.tscn")
var channel_scene : PackedScene = preload("res://Assets/DL/2D/Channel.tscn")
var line_scene : PackedScene = preload("res://Assets/GraphHandling/DrawnPath2D.tscn")
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
onready var edge_draw_timer = $EdgeDrawTimer
var weight_id_to_edge = {} # weight_id_to_edge[out_channel_ind, out_group_ind] is an edge.
var queue_draw_edges = [] # Elements look like [out_channel_ind, out_group_ind, in_channel_ind, color]
var previous_network_module_details_manager : Control setget set_previous_network_module_details_manager # Set by table

signal request_image_data(network_module_resource, feature_visualization_mode)


func _ready():
	var _error
	_error = connect("request_image_data", DLManager, "on_request_image_data")
	set_n_cols(n_cols_inspector)
	

"""
General controls
"""

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
	recreate_channels()
	update_details_layout()
	update_text()
	emit_signal("request_image_data", network_module_resource, feature_visualization_mode)
	

func recreate_channels():
	# Delete old channels:
	for child in channel_container.get_children():
		channel_container.remove_child(child)
		child.queue_free()
	
	# Create channel nodes
	for i in range(network_module_resource.size[1]):
		var new_instance = channel_scene.instance()
		channel_container.add_child(new_instance)
		# Connect signal such that we get notified when a weight is changed.
		var _error = new_instance.connect("weight_changed", self, "on_weight_changed")
	
# Called by DLManager via group
func receive_classification_results(_results):
	# When a new forward pass was performed, we need to update the image data.
	if not feature_visualization_mode and network_module_resource != null:
		emit_signal("request_image_data", network_module_resource, feature_visualization_mode)
		

func set_feature_visualization_mode(feature_visualization_mode_):
	feature_visualization_mode = feature_visualization_mode_
	if network_module_resource != null:
		# If we already have a network module resource we update the images.
		emit_signal("request_image_data", network_module_resource, feature_visualization_mode)
		
		
# Called by DLManager via group
func set_fv_image_resource(_image_resource):
	if feature_visualization_mode:
		emit_signal("request_image_data", network_module_resource, feature_visualization_mode)
	

"""
Add info text
"""


func get_size_as_string(my_array: Array):
	var my_string = "["
	var item = my_array
	while true:
		if item is Array:
			my_string = my_string + String(item.size()) + ","
			item = item[0]
		else: break
	my_string = my_string.substr(0, my_string.length()-1) + "]"
	return my_string
	
	
func add_text_weight(weight_name, weight_range):
	add_text(weight_name)
	add_text("Weights shape:")
	add_text(get_size_as_string(network_module_resource.data[weight_name]))
	add_text("Weight limits:")
	add_text(String(weight_range))
	
	
func update_text():
	for child in module_notes_container.get_children():
		child.queue_free()
	
	add_text(network_module_resource.label)
	
	if network_module_resource.size:
		add_text("Tensor size:")
		add_text(String(network_module_resource.size.slice(1,-1)))
	
	for weight_name in network_module_resource.weight_names:
		if weight_name in network_module_resource.data:
			add_text_weight(weight_name, network_module_resource.data[weight_name + "_range"])
		
	if "PFModule_kernels" in network_module_resource.data:
		add_text("Kernel size:")
		add_text(get_size_as_string(network_module_resource.data["PFModule_kernels"]))
		
		
func add_text(text):
	var new_text = text_scene.instance()
	new_text.text = text
	module_notes_container.add_child(new_text)


"""
Prepare and display image data
"""


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
		

func update_image_data(image_resources):
	set_loading(false)
	
	if not details_layout:
		# Delete old channels and add new ones. This allows the number of channels to be dynamic. We do not do this when details layout is active, because it would be too expensive.
		recreate_channels()
	
	for i in range(len(image_resources)):
		var image_resource = image_resources[i]
		var channel = channel_container.get_child(i)
		channel.set_image_resource(image_resource)
		
	# Update the legend
	var image_resource = image_resources[0]
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
	
	# Reset the vseparation of the channel_container to default value
#	channel_container.set("custom_constants/vseparation", 10)
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
#		channel_container.set("custom_constants/vseparation", 10 + (256 + label_width) * (in_channels / out_channels - 1))


"""
Toggle visibility of items and change other visuals
"""

func set_module_notes_panel_visibility(mode : bool):
	module_notes_panel_visibility = mode
	$ModuleNotesPanel.visible = mode
	

func set_legend_visible(mode : bool):
	legend_visible = mode
	$ImagePanel/VBoxContainer/Legend.visible = mode


func set_n_cols(n_cols_):
	if not is_inside_tree() or not "columns" in channel_container:
		pass
	n_cols = n_cols_
	channel_container.columns = n_cols


func _on_image_scale_changed():
	if channel_container.get_child_count() == 0:
		set_n_cols(n_cols_inspector)
		return
	
	# Change scale of image tiles.
	var scale_new = image_scale_bar.value
	var image_size = 256 * scale_new
	for child in channel_container.get_children():
		child.set_size_of_children(image_size)
	
	# Change number of rows.
	if not details_layout:
		var width_of_image_grid = 256 * n_cols_inspector
		var h_separation = channel_container.get("custom_constants/hseparation")
		var columns = int(width_of_image_grid / (image_size + h_separation))
		channel_container.columns = columns
		

"""
Draw details
"""

func set_previous_network_module_details_manager(manager):
	previous_network_module_details_manager = manager
	if previous_network_module_details_manager == null:
		clear_edges()
	else:
		prepare_edges()


func clear_edges():
	var edges = $Edges
	for child in edges.get_children():
		edges.remove_child(child)
		child.queue_free()
	padding_rect.visible = false


func extend_list(a, b):
	for b_ in b:
		a.append(b_)


func flatten_array(x):
	var elements = []
	if x[0] is Array:
		if x[0][0] is Array:
			for x_i in x:
				extend_list(elements, flatten_array(x_i))
		else:
			for x_i in x:
				for item in x_i:
					elements.append(item)
		return elements
	else:
		return x
		
		
func update_details_layout():
	
	for child in channel_container.get_children():
		child.clear_details()
		
	weight_id_to_edge.clear()
	clear_edges()
	
	if details_layout and network_module_resource != null:
		
		if "PFModule_kernels" in network_module_resource.data:
			# Draw the kernels.
			var kernels = network_module_resource.data["PFModule_kernels"]
			var i = 0
			for child in channel_container.get_children():
				var kernel_id = i % kernels.size()
				var kernel = kernels[kernel_id]
				child.draw_PFModule_kernels(kernel)
				i += 1
		if "grouped_conv_weight" in network_module_resource.data:
			# Draw the group conv.
			var weights = network_module_resource.data["grouped_conv_weight"]
			var weight_range = network_module_resource.data["grouped_conv_weight_range"]
			var i = 0
			for child in channel_container.get_children():
				var weights_group = flatten_array(weights[i])
				child.create_weights(weights_group, weight_range, "grouped_conv_weight")
				i += 1
		elif "sparse_conv_weight_selection" in network_module_resource.data:
			# Draw Sparse Conv
			var selection_weights = network_module_resource.data["sparse_conv_weight_selection"]
			var selection_weights_range = network_module_resource.data["sparse_conv_weight_selection_range"]
			var group_weights = network_module_resource.data["sparse_conv_weight_group"]
			var group_weights_range = network_module_resource.data["sparse_conv_weight_group_range"]
			var group_size = int(network_module_resource.data["group_size"])
			var i = 0
			var n_selectors = selection_weights.size()
			for child in channel_container.get_children():
				var selection_weights_channel = []
				var group_weights_channel = []
				for selector in range(n_selectors):
					selection_weights_channel.append(selection_weights[selector][i][0][0][0])
					group_weights_channel.append(group_weights[selector][0][i][0][0])
				child.create_weights(selection_weights_channel, selection_weights_range, "sparse_conv_weight_selection")
				child.create_weights(group_weights_channel, group_weights_range, "sparse_conv_weight_group")
				i += 1
		elif "blend_weight" in network_module_resource.data:
			# Blend weights
			var weights = network_module_resource.data["blend_weight"][0]
			var weight_range = network_module_resource.data["blend_weight_range"]
			var i = 0
			for child in channel_container.get_children():
				child.create_weights([weights[i][0][0]], weight_range, "blend_weight")
				i += 1
		elif "weight_per_channel" in network_module_resource.data:
			# Blend weights
			var weights = network_module_resource.data["weight_per_channel"]
			var weight_range = network_module_resource.data["weight_per_channel_range"]
			var i = 0
			for child in channel_container.get_children():
				child.create_weights([weights[i][0][0][0]], weight_range, "weight_per_channel")
				i += 1
		
		
func prepare_edges():
	if previous_network_module_details_manager == null or \
		not "draw_edges" in network_module_resource.tags or \
		$Edges.get_child_count() > 0:
		return
		
	if "grouped_conv_weight" in network_module_resource.data:
		var weights = network_module_resource.data["grouped_conv_weight"]
		var input_mapping = network_module_resource.data["input_mapping"]
		var weight_range = network_module_resource.data["grouped_conv_weight_range"]
		for out_channel in range(weights.size()):
			var group_weights = weights[out_channel]
			for group_ind in range(group_weights.size()):
				var weight = group_weights[group_ind][0][0]
				var color = ImageProcessing.get_colormap_color(weight, weight_range)
				add_to_queue_draw_edge(out_channel, group_ind, input_mapping[out_channel][group_ind], color)
				
	elif "sparse_conv_weight_selection" in network_module_resource.data:
		var out_channels = network_module_resource.size[1]
		var group_size = network_module_resource.data['group_size']
		var input_mapping = network_module_resource.data["input_mapping"]
		for out_channel in range(out_channels):
			for group_ind in range(group_size):
				add_to_queue_draw_edge(out_channel, 0, input_mapping[out_channel][group_ind])
				
	elif "input_mapping" in network_module_resource.data:
		var input_mapping = network_module_resource.data["input_mapping"]
		for i in range(input_mapping.size()):
				add_to_queue_draw_edge(i, -1, input_mapping[i])
				

func add_to_queue_draw_edge(out_channel_ind, out_group_ind, in_channel_ind, color=null):
	queue_draw_edges.append([out_channel_ind, out_group_ind, in_channel_ind, color])
	edge_draw_timer._on_trigger()
	padding_rect.visible = true
	

# Called by EdgeDrawTimer on timeout via signal
func draw_edges():
	# Draw dummy rect such that the channel connections have enough space to be shown
	for item in queue_draw_edges:
		var edge = draw_edge(item[0], item[1], item[2], item[3])
		if not item[0] in weight_id_to_edge:
			weight_id_to_edge[item[0]] = {}
		weight_id_to_edge[item[0]][item[1]] = edge
	queue_draw_edges.clear()
	
	
func draw_edge(out_channel_ind, out_group_ind, in_channel_ind, color=null):
	# get previous channel
	if previous_network_module_details_manager == null:
		return
	var previous_node = previous_network_module_details_manager.channel_container.get_child(in_channel_ind)
	if previous_node == null:
		return
	
	# If out_group_ind==-1 there is no slider
	var end_node = channel_container.get_child(out_channel_ind)
	if out_group_ind >= 0:
		end_node = end_node.get_node("Details").get_child(0).get_child(out_group_ind)
	
	var offset = rect_global_position
	var start_pos = previous_node.rect_global_position + Vector2(previous_node.rect_size.x, 0.5*previous_node.rect_size.y)
	var end_pos = end_node.rect_global_position + Vector2(0.0, 0.5*end_node.rect_size.y)
	var curve = Curve2D.new()
	curve.add_point(start_pos - rect_global_position, Vector2(0, 0), Vector2(0, 0))
	curve.add_point(end_pos - rect_global_position, Vector2(0, 0), Vector2(0, 0))
	var line_instance = line_scene.instance()
	line_instance.set_curve(curve)
	if color != null:
		line_instance.set_color(color)
	$Edges.add_child(line_instance)
	return line_instance


"""
Weight operations
"""

# Called by signal when a weight is changed by a channel node.
# TODO weight handling for different weight types
func on_weight_changed(weight, channel_ind, group_ind, weight_type):
	var weights = network_module_resource.data[weight_type]
	var adjust_weight_color = true
	
	if weight_type == "grouped_conv_weight":
		weights[channel_ind][group_ind][0][0] = weight	
	elif weight_type == "blend_weight":
		weights[0][channel_ind][0][0] = weight
	elif weight_type == "weight_per_channel":
		weights[channel_ind][0][0][0] = weight
	elif weight_type == "sparse_conv_weight_selection":
		weights[group_ind][channel_ind][0][0][0] = weight
		adjust_weight_color = false
	elif weight_type == "sparse_conv_weight_group":
		weights[group_ind][0][channel_ind][0][0] = weight
		adjust_weight_color = false
	
	if adjust_weight_color and weight_id_to_edge:
		var color = ImageProcessing.get_colormap_color(weight, network_module_resource.data[weight_type+"_range"])
		var edge = weight_id_to_edge[channel_ind][group_ind]
		edge.set_color(color)
		
	get_tree().call_group("on_weight_changed", "weight_changed", network_module_resource)


func set_elements_zero(x):
	if x[0] is Array:
		for i in x:
			set_elements_zero(i)
	else:
		for i in range(x.size()):
			x[i] = 0.0


# Called by NetworkModuleActionSelector
func zero_weights():
	for weight_name in network_module_resource.weight_names:
		if weight_name == "sparse_conv_weight_selection":
			continue
		if weight_name == "sparse_conv_weight_group":
			continue
		if weight_name in network_module_resource.data:
			var module_id_to_highlights = ArchitectureManager.channel_highlighting.module_id_to_highlights
			var highlights = module_id_to_highlights[network_module_resource.module_id]
			var weights = network_module_resource.data[weight_name]
			for channel_id in range(weights.size()):
				if highlights[channel_id] == 0.0:
					set_elements_zero(weights[channel_id])
	get_tree().call_group("on_weight_changed", "weight_changed", network_module_resource)
	update_details_layout()
	
	
# Called by NetworkModuleActionSelector
func identity_weights():
	if "grouped_conv_weight" in network_module_resource.data:
		var weights = network_module_resource.data["grouped_conv_weight"]
		var group = 0
		var group_size = weights[0].size()
		for channel_weights in weights:
			for i in range(group_size):
				if group == i:
					channel_weights[i][0][0] = 1.0
				else:
					channel_weights[i][0][0] = 0.0
			group = (group + 1) % group_size
	get_tree().call_group("on_weight_changed", "weight_changed", network_module_resource)
	update_details_layout()

"""
Loading mode
"""

func loading_fv_data(module_id):
	if feature_visualization_mode and network_module_resource != null and module_id == network_module_resource.module_id:
		for child in channel_container.get_children():
			child.queue_free()
		set_loading(true)
		
		
func set_loading(mode : bool):
	$LoadingScreen.visible = mode
	$ImagePanel.visible = not mode
