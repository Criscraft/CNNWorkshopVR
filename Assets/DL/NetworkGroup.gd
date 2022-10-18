extends PanelContainer

export var network_group_resource : Resource setget set_network_group_resource
export var highlighted_color : Color
var highlighted: bool = false setget set_highlighted

func set_network_group_resource(network_group_resource_):
	# Update network group resource
	network_group_resource = network_group_resource_
	$GridContainer/Titel/Titel.text = network_group_resource.label
	# Update collision shape
	# Give the control nodes time to update their size
	yield(get_tree(), "idle_frame")
	$SelectionBody/CollisionShape2D.position = get_rect().size / 2
	$SelectionBody/CollisionShape2D.scale = get_rect().size / 2


func set_highlighted(highlighted_):
	highlighted = highlighted_
	if highlighted:
		self_modulate = highlighted_color
	else:
		self_modulate = Color(1, 1, 1, 1)
