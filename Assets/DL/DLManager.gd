extends Node

# The URL we will connect to
export var websocket_url = "ws://localhost:8000/"

# Our WebSocketClient instance
var _client = WebSocketClient.new()

onready var my_thread_pool_manager = $MyThreadPoolManager

signal receive_dataset_images(image_resources)
signal on_connected()

func _ready():
	# Connect base signals to get notified of connection open, close, and errors.
	_client.connect("connection_closed", self, "_closed")
	_client.connect("connection_error", self, "_closed")
	_client.connect("connection_established", self, "_connected")
	# This signal is emitted when not using the Multiplayer API every time
	# a full packet is received.
	# Alternatively, you could check get_peer(1).get_available_packets() in a loop.
	_client.connect("data_received", self, "_on_data")

	# Initiate connection to the given URL.
	var err = _client.connect_to_url(websocket_url, []) # No subprotocoll used yet.
	if err != OK:
		print("Unable to connect")
		set_process(false)
		
	$MyThreadPoolManager.pool.connect("task_completed", self, "_on_task_completed")

func _closed(was_clean = false):
	# was_clean will tell you if the disconnection was correctly notified
	# by the remote peer before closing the socket.
	print("Closed, clean: ", was_clean)
	set_process(false)

func _connected(proto = ""):
	# This is called on connection, "proto" will be the selected WebSocket
	# sub-protocol (which is optional)
	print("Connected with protocol: ", proto)
	# You MUST always use get_peer(1).put_packet to send data to server,
	# and not put_packet directly when not using the MultiplayerAPI.
	# _client.get_peer(1).put_packet("Test packet".to_utf8())
	emit_signal("on_connected")

func _on_data():
	# Print the received packet, you MUST always use get_peer(1).get_packet
	# to receive data from server, and not get_packet directly when not
	# using the MultiplayerAPI.
	#print("Got data from server: ", _client.get_peer(1).get_packet().get_string_from_utf8())
	var data = _client.get_peer(1).get_packet().get_string_from_utf8()
	data = JSON.parse(data).result
	match data["resource"]:
		
		"send_dataset_images":
			print("DLManager reveived dataset images.")
			my_thread_pool_manager.submit_task(self, "preprocess_dataset_images", data["image_resources"], "preprocess_dataset_images")


func preprocess_dataset_images(image_resource_data):
	var image_resources = []
	for item in image_resource_data:
		image_resources.append(dict_to_image_resource(item))
	return image_resources


func _on_task_completed(task):
	match task.tag:
		
		"preprocess_dataset_images":
			call_deferred("emit_signal", "receive_dataset_images", task.result)
		

func _on_DatasetManager_get_dataset_images(ids):
	var message = {"resource" : "get_dataset_images", "ids" : ids}
	message = JSON.print(message)
	print(message)
	_client.get_peer(1).put_packet(message.to_utf8())

func dict_to_image_resource(dick : Dictionary):
	var image_resource = DLImageResource.new()
	image_resource.mode = DLImageResource.MODE[dick["mode"]]
	#var pool_byte_array = Marshalls.base64_to_raw(dick["data"])
	var image = Image.new()
	image.load_png_from_buffer(Marshalls.base64_to_raw(dick["data"]))
	image_resource.image = image
	image_resource.id = dick["id"]
	image_resource.label = dick["label"]
	return image_resource

func _process(_delta):
	# Call this in _process or _physics_process. Data transfer, and signals
	# emission will only happen when calling this function.
	_client.poll()

