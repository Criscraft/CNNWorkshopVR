extends Spatial

export var image_resource : Resource setget setFunction

func setFunction(image_resource_):
	image_resource = image_resource_
	# Update the displayed image and the label. If this node is being duplicated and the copy is not yet added to the scene tree, we skip.
	if get_tree():
		update_image()
		update_label()
	
func update_image() -> void:
	var image_texture = ImageTexture.new()
	image_texture.create_from_image(image_resource.image, 0)
	var material = SpatialMaterial.new()
	material.set_texture(SpatialMaterial.TEXTURE_ALBEDO, image_texture)
	material.set_flag(SpatialMaterial.FLAG_UNSHADED, true)
	$Image.set_surface_material(0, material)
	
func update_label() -> void:
	pass
