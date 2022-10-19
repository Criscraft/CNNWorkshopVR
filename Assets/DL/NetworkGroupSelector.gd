extends Area2D

var last_selected_node
export var target_group : String
export var method_to_be_called_in_group : String


func _on_Selector_body_entered(_body):
	update_selection()

func _on_Selector_body_exited(_body):
	update_selection()

func update_selection():
	var bodies = get_overlapping_bodies()
	if not bodies:
		return
	var closest_distance = 1e9
	var closest_body
	var new_distance
	
	for body in bodies:
		if not "highlighted" in body.get_parent():
			continue
		new_distance = body.global_position.distance_to(global_position)
		if new_distance < closest_distance:
			closest_distance = new_distance
			closest_body = body
	
	if closest_body != null:
		if is_instance_valid(last_selected_node):
			last_selected_node.highlighted = false
		last_selected_node = closest_body.get_parent()
		last_selected_node.highlighted = true
		#get_tree().call_group("on_network_group_selected_by_overvie_screen", "network_group_selected_by_overvie_screen", network_group_resource)
		get_tree().call_group(target_group, method_to_be_called_in_group, last_selected_node)
		
