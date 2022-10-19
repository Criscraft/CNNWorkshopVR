extends PanelContainer
class_name MyGraphNode

export var highlighted_color : Color
var highlighted: bool = false setget set_highlighted
var id : int
var precursors : Array

func set_text(text):
	$GridContainer/Titel/Titel.text = text
	# Update collision shape
	# Give the control nodes time to update their size
	if is_inside_tree():
		# maybe use update?
		yield(get_tree(), "idle_frame")
	$SelectionBody/CollisionShape2D.position = get_rect().size / 2
	$SelectionBody/CollisionShape2D.scale = get_rect().size / 2


func set_highlighted(highlighted_):
	highlighted = highlighted_
	if highlighted:
		self_modulate = highlighted_color
	else:
		self_modulate = Color(1, 1, 1, 1)
