from godot import exposed, export, Node, Image, PoolByteArray, ResourceLoader, GDScript
import numpy as np


@exposed
class PythonDataset(Node):

	@export(str)
	@property
	def data_path(self):
		return self._data_path
	
	@data_path.setter
	def data_path(self, value):
		self._data_path = value
		self.setup_pytorch_dataset()

	def _ready(self):
		"""
		Called every time the node is added to the scene.
		Initialization here.
		"""
		# Init Pytorch dataset
		pass
		
	def setup_pytorch_dataset(self):
		pass
		
	def get_image(self, i):
		h = 8
		w = 12
		
		#with open('b.txt', 'w') as f:
		#	f.write(str(dir(GDScript)))
		
		image = np.random.random((3, h, w))
		# Godot Image code https://github.com/godotengine/godot/blob/7610409b8a14b8499763efa76578795c755a846d/core/image.cpp#L2613-L2618
		# Godot needs different ordering of the dimensions: (h, w, c)
		image[0,0,6] = 0.
		image[1,0,6] = 0.
		image[2,0,6] = 0.
		image = image.transpose((1,2,0))
		image = image.flatten()
		image = image * 255
		image = image.astype(np.uint8)
		image_bytes = image.tobytes()
		arr = PoolByteArray() # create the array
		arr.resize(len(image_bytes))
		
		with arr.raw_access() as ptr:
			for i in range(len(image_bytes)):
				ptr[i] = image_bytes[i] # this is fast
				
		image_out = Image()
		image_out.create_from_data(w, h, False, 4, arr)
		
		image_resource = ResourceLoader.load("res://Assets/DL/DLImageResource.gd")
		resource = GDScript()
		resource.set_script(image_resource)
		resource.set("image", image_out)
		resource.set("mode", resource.get("MODE").get("DATASET"))
		resource.set("h", h)
		resource.set("w", w)
		return resource
		
		
