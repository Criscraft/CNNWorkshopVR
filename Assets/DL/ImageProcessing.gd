extends Script
class_name ImageProcessing

static func dict_to_image_resource(dick : Dictionary):
	var image_resource = DLImageResource.new()
	image_resource.mode = DLImageResource.MODE[dick["mode"]]
	#var pool_byte_array = Marshalls.base64_to_raw(dick["data"])
	var image = Image.new()
	image.load_png_from_buffer(Marshalls.base64_to_raw(dick["data"]))
	image_resource.image = image
	image_resource.id = dick["id"]
	image_resource.label = dick["label"]
	image_resource.value_zero_decoded = dick["value_zero_decoded"]
	image_resource.value_255_decoded = dick["value_255_decoded"]
	return image_resource
