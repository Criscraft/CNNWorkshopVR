extends Script
class_name ImageProcessing

static func dict_to_image_resource(dick : Dictionary):
	var image_resource = ImageResource.new()
	image_resource.mode = ImageResource.MODE[dick["mode"]]
	#var pool_byte_array = Marshalls.base64_to_raw(dick["data"])
	image_resource.image = get_image_from_raw(dick["data"])
	image_resource.id = dick["id"]
	image_resource.label = dick["label"]
	image_resource.module_id = dick["module_id"]
	image_resource.channel_id = dick["channel_id"]
	image_resource.value_zero_decoded = dick["value_zero_decoded"]
	image_resource.value_255_decoded = dick["value_255_decoded"]
	return image_resource


static func array_to_grayscaleimage(array : Array):
	var image = Image.new()
	var width = array[0].size()
	var height = array.size()
	image.create(width, height, false, 4) 
	image.lock()
	var value 
	for i in range(height):
		for j in range(width):
			value = array[i][j]
			image.set_pixel(j, i, Color(value, 0, -value))
	image.unlock()
	return image


static func get_colormap_color(value, value_range=[0.0, 1.0]):
	var color : Color
	if value < 0:
		color = Color(0, 0, value / (value_range[0] + 1e-6))
	else:
		color = Color(value / (value_range[1] + 1e-6), 0, 0)
	return color


static func get_image_from_raw(utf8):
	var image = Image.new()
	image.load_png_from_buffer(Marshalls.base64_to_raw(utf8))
	return image
