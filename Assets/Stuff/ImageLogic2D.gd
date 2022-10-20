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
