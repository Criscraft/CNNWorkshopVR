extends Spatial

export var network_module_details_screen2D_scene : PackedScene = preload("res://Assets/DL/2D/NetworkModuleDetailsScreen2D.tscn")
export var image_tile_scene : PackedScene = preload("res://Assets/DL/2D/ImageTile.tscn")
export var fv_settings_screen_scene : PackedScene = preload("res://Assets/DL/2D/FVSettingsScreen.tscn")

var layout_details_mode = true
var feature_visualization_mode = false
var screen_slot : Control
var previous_screen_slot : Control
var next_screen_slot : Control

onready var network_module_action_selector = $NetworkModuleActionSelector
	

func _on_Snap_Zone_has_picked_up(pickable_object):
	var module_logic = pickable_object.get_node_or_null("ModuleLogic")
	if is_instance_valid(module_logic):
		# A pickable module was placed into the snap tray
		on_receive_network_module_resource(module_logic.network_module_resource)
		return
	
	var image_logic = pickable_object.get_node_or_null("ImageLogic")
	if is_instance_valid(image_logic):
		# A pickable image was placed into the snap tray
		on_receive_image_resource(image_logic.image_resource)
		return
		
	var fv_settings = pickable_object.get_node_or_null("FVSettings")
	if is_instance_valid(fv_settings):
		# A pickable feature visualization setting was placed into the snap tray
		on_receive_fv_settings_resource(fv_settings.fv_settings_resource)
		return


func _on_Snap_Zone_has_dropped():
	for child in screen_slot.get_children():
		child.queue_free()
	if next_screen_slot != null:
		var next_manager = next_screen_slot.get_node_or_null("NetworkModuleDetailsScreen2D/NetworkModuleDetailsManager")
		if next_manager != null:
			next_manager.previous_network_module_details_manager = null
	
	
func on_receive_network_module_resource(network_module_resource):
	# Note that when lever_switch_layout_status is true we want to see details
	# Note that when feature_visualization_mode is false we want to show activations
	if network_module_resource != null:
		screen_slot.add_child(network_module_details_screen2D_scene.instance())
		var network_module_details_manager = get_network_module_details_manager()
		network_module_details_manager.feature_visualization_mode = feature_visualization_mode
		network_module_details_manager.details_layout = layout_details_mode
		network_module_details_manager.network_module_resource = network_module_resource
		# By default the ModuleNotesPanel should be invisible to save space.
		network_module_details_manager.module_notes_panel_visibility = false
		
		network_module_action_selector.network_module_details_manager = network_module_details_manager
		
		if previous_screen_slot != null:
			var previous_manager = previous_screen_slot.get_node_or_null("NetworkModuleDetailsScreen2D/NetworkModuleDetailsManager")
			if previous_manager != null:
				network_module_details_manager.previous_network_module_details_manager = previous_manager
	else:
		network_module_action_selector.network_module_details_manager = null
	

func on_receive_image_resource(image_resource):
	if image_resource != null:
		var image_tile = image_tile_scene.instance()
		screen_slot.add_child(image_tile)
		image_tile.image_resource = image_resource
		image_tile.set_size_of_children(512)


func on_receive_fv_settings_resource(fv_settings_resource):
	if fv_settings_resource != null:
		var new_instance = fv_settings_screen_scene.instance()
		screen_slot.add_child(new_instance)
		new_instance.fv_settings_resource = fv_settings_resource


func _on_lever_switch_layout_status_change(status_):
	layout_details_mode = status_
	if is_inside_tree():
		update_details_layout()
		
		
func update_details_layout():
	# Note that when lever_switch_layout_details is true we want to see details
	var network_module_details_manager = get_network_module_details_manager()
	if network_module_details_manager != null:
		network_module_details_manager.details_layout = layout_details_mode
	

func _on_lever_switch_activation_fv_status_change(status_):
	feature_visualization_mode = not status_
	if is_inside_tree():
		update_feature_visualization_mode()
		
		
func update_feature_visualization_mode():
	# Note that when status_ is true we want to show activations, not feature visualizations
	var network_module_details_manager = get_network_module_details_manager()
	if network_module_details_manager != null:
		network_module_details_manager.feature_visualization_mode = feature_visualization_mode


func _on_ToggleLayout_pressed(_button):
	var network_module_details_manager = get_network_module_details_manager()
	if network_module_details_manager != null:
		network_module_details_manager.module_notes_panel_visibility = not network_module_details_manager.module_notes_panel_visibility
		
		
func get_network_module_details_manager():
	if screen_slot.get_child_count() > 0:
		var network_module_details_manager = screen_slot.get_node("NetworkModuleDetailsScreen2D/NetworkModuleDetailsManager")
		return network_module_details_manager
	return null
