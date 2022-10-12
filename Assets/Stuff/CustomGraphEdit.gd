extends Node2D

export var x_margin : int = 50
export var y_margin : int = 100
export var drawn_path_szene : PackedScene = preload("res://Assets/Stuff/DrawnPath2D.tscn")
var id_to_child_dict = {} # Is not updated automatically
var layering = [] # Is not updated automatically
#func _process(delta):
#	if Input.is_action_just_pressed("debug_action"):
#		arrange_nodes()

func arrange_nodes():
	var precursors = {}
	id_to_child_dict = {}
	for node in get_children():
		if "network_group_resource" in node:
			precursors[node.network_group_resource.id] = node.network_group_resource.precursors
			id_to_child_dict[node.network_group_resource.id] = node
	if not id_to_child_dict:
		return
	
	layering = get_layering(id_to_child_dict.keys(), precursors)
	layering = cross_minimization(layering)
	position_nodes()
	#get_viewport().update_worlds()
	#call_deferred("draw_edges")
	draw_edges()
	

func get_layering(node_ids, precursors):
	var q = node_ids # Total set of nodes
	var u = [] # Set of nodes that already have been assigned to layers.
	var p = node_ids.duplicate() # tracks q - u
	var z = [] # Set of nodes that are precursors of the current layer
	var current_layer_nodes = []
	var selected = false
	layering = []
	
	while not SetOperations.is_equal(q, u):
		# Update set of nodes which are still unassigned.
		p = SetOperations.difference(p, u)
		for p_ in p:
			if SetOperations.is_subset(precursors[p_], z):
				# The node p_ has all of its precursors in z, the set of nodes that are precursors of the current layer.
				# Add the node to the current layer.
				current_layer_nodes.append(p_)
				selected = true
				u.append(p_)
		# If we did not add any node to the current layer, move on to the next layer.
		if not selected:
			layering.append(current_layer_nodes)
			current_layer_nodes = []
			var previous_size = z.size()
			z = SetOperations.union(z, u)
			if previous_size == z.size():
				print("The graph contains a loop and will not be displayed correctly.")
				break
		selected = false
	# Append the last layer, because it has not been done yet.
	layering.append(current_layer_nodes)
	return layering
	
	
func cross_minimization(layer_dict):
	# TODO: implement cross_minimization
	return layer_dict


func get_max_sizes_per_layer():
	var x_sizes_in_one_layer = []
	var y_sizes_in_one_layer = []
	var max_x_sizes = []
	var max_y_sizes = []
	for layer in layering:
		x_sizes_in_one_layer = []
		y_sizes_in_one_layer = []
		for id in layer:
			x_sizes_in_one_layer.append(id_to_child_dict[id].rect_size.x)
			y_sizes_in_one_layer.append(id_to_child_dict[id].rect_size.y)
		max_x_sizes.append(x_sizes_in_one_layer.max())
		max_y_sizes.append(y_sizes_in_one_layer.max())
	return {"max_x_sizes" : max_x_sizes, "max_y_sizes" : max_y_sizes}
	

func position_nodes():
	#var general_node_offset = Vector2(0, max_sizes_per_layer["max_y_sizes"].max() * 0.5)
	var max_sizes_per_layer = get_max_sizes_per_layer()
	
	var last_x_position = 0
	var last_y_position = 0
	var new_x_position = 0
	var new_y_position = 0
	var y_center = 0
	var nodes_in_layer
	var node
	
	for i in range(len(layering)):
		nodes_in_layer = layering[i]
		# Choose x position of current layer
		new_x_position = last_x_position
		if i > 0:
			new_x_position += max_sizes_per_layer["max_x_sizes"][i - 1] + x_margin
		last_x_position = new_x_position
		# Choose y offset
		y_center = 0
		for node_ in nodes_in_layer:
			y_center += id_to_child_dict[node_].rect_size.y
		y_center += y_margin * (len(nodes_in_layer) - 1)
		y_center *= -0.5
		# Choose y position for each node in layer.
		last_y_position = y_center
		for j in range(len(nodes_in_layer)):
			new_y_position = last_y_position
			if j > 0:
				node = id_to_child_dict[nodes_in_layer[j - 1]]
				new_y_position += node.rect_size.y + j * y_margin
			# Position the node
			node = id_to_child_dict[nodes_in_layer[j]]
			node.rect_position = Vector2(new_x_position, new_y_position)
			last_y_position = new_y_position
			
			
func draw_edges():
	var precursor_node
	var curve
	var target_socket
	var target_position
	var precursor_right_socket
	var precursor_right_socket_position
	var new_drawn_path_szene
	var in_curve
	var epsilon = y_margin * 0.5
	
	for node in id_to_child_dict.values():
		for precursor_node_id in node.network_group_resource.precursors:
			precursor_node = id_to_child_dict[precursor_node_id]
			# if the precursor node has smaller y, we will use the top socket
			if precursor_node.rect_position.y < node.rect_position.y - epsilon:
				target_socket = node.get_node("GridContainer/Top/Socket")
				target_position = target_socket.get_global_rect().position
				in_curve = Vector2(0, -y_margin * 0.5)
			elif precursor_node.rect_position.y > node.rect_position.y + epsilon:
				target_socket = node.get_node("GridContainer/Bottom/Socket")
				target_position = target_socket.get_global_rect().position
				in_curve = Vector2(0, y_margin * 0.5)
			else:
				target_socket = node.get_node("GridContainer/Left/Socket")
				target_position = target_socket.get_global_rect().position
				in_curve = Vector2(-x_margin, 0)
			
			precursor_right_socket = id_to_child_dict[precursor_node_id].get_node("GridContainer/Right/Socket")
			precursor_right_socket_position = precursor_right_socket.get_global_rect().position
			curve = Curve2D.new()
			curve.add_point(Vector2(precursor_right_socket_position), Vector2(0, 0), Vector2(x_margin, 0))
			curve.add_point(Vector2(target_position), in_curve, Vector2(0, 0))
			
			new_drawn_path_szene = drawn_path_szene.instance()
			add_child(new_drawn_path_szene)
			new_drawn_path_szene.set_curve(curve)
	
	
class SetOperations:

	static func is_equal(a, b):
		for a_ in a:
			if not a_ in b:
				return false
		return a.size() == b.size()
		
	static func difference(a, b):
		var c = []
		for a_ in a:
			if not a_ in b:
				c.append(a_)
		return c
		
	static func is_subset(a, b):
		if not a and not b:
			return true
		for a_ in a:
			if ! a_ in b:
				return false
		return true
		
	static func union(a, b):
		var c = a.duplicate()
		for b_ in b:
			if not b_ in a:
				c.append(b_)
		return c
		
