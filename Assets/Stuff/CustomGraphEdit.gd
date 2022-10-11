extends Node2D

export var x_margin : int = 50
export var y_margin : int = 50

#func _process(delta):
#	if Input.is_action_just_pressed("debug_action"):
#		arrange_nodes()

func arrange_nodes():
	var precursors = {}
	var id_to_child_dict = {}
	for node in get_children():
		if "network_group_resource" in node:
			precursors[node.network_group_resource.id] = node.network_group_resource.precursors
			id_to_child_dict[node.network_group_resource.id] = node
	if not id_to_child_dict:
		return
	
	var layering = get_layering(id_to_child_dict.keys(), precursors)
	layering = cross_minimization(layering)
	print(layering)
	var max_sizes_per_layer = get_max_sizes_per_layer(id_to_child_dict, layering)
	position_nodes(id_to_child_dict, layering, max_sizes_per_layer)
	

func get_layering(nodes, precursors):
	var q = nodes # Total set of nodes
	var u = [] # Set of nodes that already have been assigned to layers.
	var p = nodes.duplicate() # tracks q - u
	var z = [] # Set of nodes that are precursors of the current layer
	var current_layer_nodes = []
	var selected = false
	var layering = []
	
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
	return layering
	
	
func cross_minimization(layer_dict):
	# TODO: implement cross_minimization
	return layer_dict


func get_max_sizes_per_layer(id_to_child_dict, layering):
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
	

func position_nodes(id_to_child_dict, layering, max_sizes_per_layer):
	var general_node_offset = Vector2(0, max_sizes_per_layer["max_y_sizes"].max() * 0.5)
	var last_x_position = 0
	var last_y_position = 0
	var new_x_position = 0
	for layer in layering:
		new_x_position = max_sizes_per_layer["max_x_sizes"][layer - 1] + x_margin
		for node in layer:
			new_y_position = ...
			node.rect.position = Vector2(new_x_position, new_y_position)
	
	
	
	
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
		
