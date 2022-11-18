extends Spatial

export var network_module_details_screen2D_scene : PackedScene = preload("res://Assets/DL/2D/NetworkModuleDetailsScreen2D.tscn")
export var image_tile_scene : PackedScene = preload("res://Assets/DL/2D/ImageTile.tscn")

var network_module_resource : Resource setget set_network_module_resource
var image_resource : Resource setget set_image_resource
var layout_details_mode = true
var feature_visualization_mode = false
var screen_slot : Control

onready var network_module_action_selector = $NetworkModuleActionSelector
	

func _on_Snap_Zone_has_picked_up(pickable_object):
	var module_logic = pickable_object.get_node_or_null("ModuleLogic")
	if is_instance_valid(module_logic):
		# A pickable module was placed into the snap tray
		set_network_module_resource(module_logic.network_module_resource)
		return
	
	var image_logic = pickable_object.get_node_or_null("ImageLogic")
	if is_instance_valid(image_logic):
		# A pickable image was placed into the snap tray
		set_image_resource(image_logic.image_resource)
		return


func _on_Snap_Zone_has_dropped():
	set_network_module_resource(null)
	set_image_resource(null)
	
	
func set_network_module_resource(network_module_resource_):
	if network_module_resource == network_module_resource_:
		return
		
	network_module_resource = network_module_resource_
	
	# Note that when lever_switch_layout_status is true we want to see details
	# Note that when feature_visualization_mode is false we want to show activations
	for child in screen_slot.get_children():
		child.queue_free()
	if network_module_resource != null:
		screen_slot.add_child(network_module_details_screen2D_scene.instance())
		var network_module_details_manager = get_network_module_details_manager()
		network_module_details_manager.feature_visualization_mode = feature_visualization_mode
		network_module_details_manager.details_layout = layout_details_mode
		network_module_details_manager.network_module_resource = network_module_resource
		# By default the ModuleNotesPanel should be invisible to save space.
		network_module_details_manager.module_notes_panel_visibility = false
		
		network_module_action_selector.network_module_details_manager = network_module_details_manager
	else:
		network_module_action_selector.network_module_details_manager = null
	

func set_image_resource(image_resource_):
	if image_resource == image_resource_:
		return
	image_resource = image_resource_
	for child in screen_slot.get_children():
		child.queue_free()
	if image_resource != null:
		var image_tile = image_tile_scene.instance()
		screen_slot.add_child(image_tile)
		image_tile.image_resource = image_resource
		image_tile.set_size_of_children(512)


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
