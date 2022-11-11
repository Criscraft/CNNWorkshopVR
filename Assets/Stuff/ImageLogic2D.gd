extends Control

onready var image = get_node("Image")
onready var label = get_node("Label")

export var image_resource : Resource setget set_image_resource

func set_image_resource(image_resource_):
	image_resource = image_resource_
	update_image()
	update_label()
	
func update_image() -> void:
	var image_texture = ImageTexture.new()
	image_texture.create_from_image(image_resource.image, 0)
	image.texture = image_texture
	
func update_label() -> void:
	label.text = image_resource.label
	if image_resource.label:
		label.visible = true
	else:
		label.visible = false

func set_size_of_children(size_new):
	$Image.rect_min_size = Vector2(size_new, size_new)
	$Label.rect_min_size = Vector2(size_new, 40)


func _on_ImageSelectButton_pressed():
	get_tree().call_group("on_image_selected", "image_selected", image_resource)
