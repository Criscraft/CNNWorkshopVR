extends Spatial

onready var action_label = $ActionLabel

var network_module_details_manager setget set_network_module_details_manager
var all_actions = []
var available_actions = []
var active_action_ind : int = 0 setget set_active_action_ind

signal zero_weights
signal identity_weights

class Action:
	var label : String = "Action"
	var requirements : Array = []
	var action : String = "" # corresponds to the name of a signal
	

func _ready():
	var action
	
	action = Action.new()
	action.label = "Choose action."
	action.requirements = []
	all_actions.append(action)
	
	action = Action.new()
	action.label = "zero non-marked"
	action.requirements = []
	action.action = "zero_weights"
	all_actions.append(action)
	
	action = Action.new()
	action.label = "ident. weights"
	action.requirements = ["grouped_conv_weight"]
	action.action = "identity_weights"
	all_actions.append(action)
	
	set_network_module_details_manager(null)


func set_network_module_details_manager(network_module_details_manager_):
	network_module_details_manager = network_module_details_manager_
	
	available_actions.clear()
	
	if network_module_details_manager == null:
		action_label.text = "Insert module."
		return
		
	# Check which actions are available.
	var requirements_fulfilled
	var network_module_resource = network_module_details_manager.network_module_resource
	for action in all_actions:
		requirements_fulfilled = true
		for requirement in action.requirements:
			if not requirement in network_module_resource.data:
				requirements_fulfilled = false
				break
		if requirements_fulfilled:
			available_actions.append(action)
	
	# Show the first action.
	active_action_ind = 0
	set_active_action_ind(active_action_ind)
	
	var _error
	_error = connect("zero_weights", network_module_details_manager, "zero_weights")
	_error = connect("identity_weights", network_module_details_manager, "identity_weights")
	

func set_active_action_ind(ind):
	if available_actions.size() == 0:
		return
	ind = ind % available_actions.size()
	active_action_ind = ind
	action_label.text = available_actions[active_action_ind].label
	
	

func increment_action(_collided_at=null):
	set_active_action_ind(active_action_ind + 1)
	
	
func decrement_action(_collided_at=null):
	set_active_action_ind(active_action_ind - 1)
	
	
func perform_action(_collided_at=null):
	if available_actions[active_action_ind].action:
		emit_signal(available_actions[active_action_ind].action)
	
