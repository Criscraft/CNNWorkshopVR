extends Control

onready var image = $Image
onready var label = $Label
onready var highlight_rect = $Image/HighlightRect
export var image_resource : Resource setget set_image_resource
var highlight : float = 0.0 setget set_highlight

func set_image_resource(image_resource_):
	image_resource = image_resource_
	update_image()
	update_label()
	
	
func update_image() -> void:
	var image_texture = ImageTexture.new()
	image_texture.create_from_image(image_resource.image, 0)
	image.texture = image_texture
	update_highlights()
	
	
func update_label() -> void:
	label.text = image_resource.label
	if image_resource.label:
		label.visible = true
		$Image.rect_min_size = $Image.rect_min_size - Vector2($Label.rect_min_size.y, $Label.rect_min_size.y)
	else:
		label.visible = false


func set_size_of_children(size_new):
	$Image.rect_min_size = Vector2(size_new, size_new)
	$Label.rect_min_size = Vector2(size_new, $Label.rect_min_size.y)


func _on_ImageSelectButton_pressed():
	get_tree().call_group("on_image_selected", "image_selected", image_resource)


# Called by ChannelHighlighting via group on_update_highlights.
func update_highlights():
	if image_resource == null:
		return
	if image_resource.mode != ImageResource.MODE.ACTIVATION and image_resource.mode != ImageResource.MODE.FEATURE_VISUALIZATION:
		return
		
	var module_id_to_highlights = ArchitectureManager.channel_highlighting.module_id_to_highlights
	var channel_highlights = module_id_to_highlights[image_resource.module_id]
	if not channel_highlights:
		set_highlight(0.0)
		return
	
	var highlight_new = channel_highlights[image_resource.channel_id]
	set_highlight(highlight_new)


func set_highlight(highlight_new):
	highlight = highlight_new
	if highlight == 0.0:
		highlight_rect.visible = false
	else:
		highlight_rect.visible = true
		highlight_rect.border_color = ImageProcessing.get_weight_color(highlight)
