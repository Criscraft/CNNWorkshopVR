extends Node

# The URL we will connect to
export var websocket_url = "ws://127.0.0.1:8000/"

# Our WebSocketClient instance
var _client = WebSocketClient.new()
var network_module_resources_weights_changed = []

signal on_connected()
signal on_receive_dataset_images(image_resource_data)
signal on_receive_noise_image(image_resource)

"""
#################################
Establish connection.
#################################
"""

func _ready():
	# Connect base signals to get notified of connection open, close, and errors.
	_client.connect("connection_closed", self, "_closed")
	_client.connect("connection_error", self, "_closed")
	_client.connect("connection_established", self, "_connected")
	# This signal is emitted when not using the Multiplayer API every time
	# a full packet is received.
	# Alternatively, you could check get_peer(1).get_available_packets() in a loop.
	_client.connect("data_received", self, "_on_data")
	
	# Add debouncing timer for reacting on changes in network weights
	var timer = Timer.new()
	timer.name = "NetworkWeightsDebouncingTimer"
	timer.one_shot = true
	timer.wait_time = 2
	timer.set_script(load("res://Assets/Stuff/DebouncingTimer.gd"))
	add_child(timer)
	timer.connect("timeout", self, "send_network_weights")
	add_to_group("on_weight_changed")

	# Initiate connection to the given URL.
	var err = _client.connect_to_url(websocket_url, []) # No subprotocoll used yet.
	if err != OK:
		print("Unable to connect")
		set_process(false)


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
	request_architecture()
	request_noise_image()
	
	
func _process(_delta):
	# Call this in _process or _physics_process. Data transfer, and signals
	# emission will only happen when calling this function.
	_client.poll()
	

"""
#################################
Receive data
#################################
"""

func _on_data():
	# Print the received packet, you MUST always use get_peer(1).get_packet
	# to receive data from server, and not get_packet directly when not
	# using the MultiplayerAPI.
	#print("Got data from server: ", _client.get_peer(1).get_packet().get_string_from_utf8())
	var data = _client.get_peer(1).get_packet().get_string_from_utf8()
	data = JSON.parse(data).result
	match data["resource"]:
		
		"request_dataset_images":
			print("DLManager reveived dataset images.")
			emit_signal("on_receive_dataset_images", data["image_resources"])
			
		"request_forward_pass":
			print("DLManager reveived classification results.")
			get_tree().call_group("on_receive_classification_results", "receive_classification_results", data["results"])
			
		"request_architecture":
			print("DLManager reveived architecture.")
			ArchitectureManager.create_network_group_resources(data["architecture"]["group_dict"])
			ArchitectureManager.create_network_module_resources(data["architecture"]["module_dict"])
			
		"request_image_data":
			print("DLManager reveived image data.")
			get_tree().call_group("on_receive_image_data", "receive_image_data", data["image_resources"])
			
		"request_noise_image":
			print("DLManager reveived noise image.")
			emit_signal("on_receive_noise_image", data["image_resource"])
			
		_:
			print("No match in DLManager.")


"""
#################################
Request data
#################################
"""

# called by signal from Image Shelf
func on_request_dataset_images(n):
	var message = {"resource" : "request_dataset_images", "n" : n}
	send_request(message)
	
	
# called by signal from Network Station
func on_request_forward_pass(image_resource : Dictionary):
	var message = {"resource" : "request_forward_pass", "image_resource" : image_resource}
	send_request(message)
	
	
# Called in _ready()
func request_architecture():
	var message = {"resource" : "request_architecture"}
	send_request(message)
	
	
# Called by signal from NetworkModuleDetailsManager
func on_request_image_data(network_module_resource : NetworkModuleResource, feature_visualization_mode=false):
	var message = {"resource" : "request_image_data", "network_module_resource" : network_module_resource.get_dict()}
	if feature_visualization_mode:
		message["mode"] = "fv"
		# Inform that we load feature visualization data, which might require time.
		get_tree().call_group("on_loading_fv_data", "loading_fv_data", network_module_resource.module_id)
	else:
		message["mode"] = "activation"
	send_request(message)


func send_network_weights():
	if not network_module_resources_weights_changed:
		return
	var weight_dicts = {}
	for network_module_resource in network_module_resources_weights_changed:
		weight_dicts[network_module_resource.module_id] = network_module_resource.weights
	var message = {"resource" : "request_network_weights", "weight_dicts" : weight_dicts}
	network_module_resources_weights_changed = []
	send_request(message)
	
	
func send_request(request_dictionary : Dictionary):
	var message = JSON.print(request_dictionary)
	if message.length() < 100:
		print(message)
	_client.get_peer(1).put_packet(message.to_utf8())
	

# called by signal from Network Station
func on_set_fv_image_resource(image_resource : Dictionary):
	var message = {"resource" : "set_fv_image_resource", "image_resource" : image_resource}
	send_request(message)
	# Inform group that the fv image resource is updated.
	get_tree().call_group("on_set_fv_image_resource", "set_fv_image_resource", image_resource)


# Called in _ready()
func request_noise_image():
	var message = {"resource" : "request_noise_image"}
	send_request(message)

"""
#################################
Submit changes invoced by client
#################################
"""

# Called by NetworkModuleDetailsManager via group
func weight_changed(network_module_resource):
	if not network_module_resource in network_module_resources_weights_changed:
		network_module_resources_weights_changed.append(network_module_resource)
	$NetworkWeightsDebouncingTimer._on_trigger()
