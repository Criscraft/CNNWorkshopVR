extends Node
# initialization parameters 
onready var pool = FutureThreadPool.new()
onready var wait = Mutex.new()

# setting parameters
var thread_count = 1 # amount of threads in thread pool
var no_timer_thread:bool = false # note if enabled task_time_limit will not work as it depends on the timer thread to actually cancel tasks
var task_time_limit:float = 100000 # in milliseconds
var default_priority: int = 100

# initialization phase
func _ready():
	__start_pool()
	pool.connect("task_completed", self, "on_task_completed")

func on_task_completed(task):
	get_tree().call_group("on_pool_task_completed", "on_pool_task_completed", task)

func __start_pool():
	pool.__thread_count = thread_count
	pool.no_timer_thread = no_timer_thread
	pool.__pool = pool.__create_pool()
	
# post initialization phase
func join(identifier, by: String = "task"):
	return pool.join(identifier, by)

func get_task_queue_as_immutable(immutable: bool = true):
	if immutable:
		return pool.__tasks.duplicate(false)
	else:
		return pool.__tasks

func get_pending_queue_as_immutable(immutable: bool = true):
	if immutable:
		return pool.__pending.duplicate(false)
	else:
		return pool.__pending

func get_threads_as_immutable(immutable: bool = true): # should only really be used for debugging
	if immutable:
		return pool.__pool.duplicate(false)
	else:
		print("getting_threads_when_not_immutable_is_a_really_bad_idea")
		return pool.__pool

func submit_task(instance: Object, method: String, parameter,task_tag : String ,time_limit : float = task_time_limit, priority:int = default_priority):
	return pool.submit_task(instance, method, parameter,task_tag, time_limit, priority)

func submit_task_as_parameter(instance: Object, method: String, parameter,task_tag : String, time_limit : float = task_time_limit, priority:int = default_priority):
	return pool.submit_task_as_parameter(instance, method, parameter ,task_tag, time_limit, priority)

func submit_task_unparameterized(instance: Object, method: String, task_tag : String, time_limit : float = task_time_limit, priority:int = default_priority):
	return pool.submit_task_unparameterized(instance, method ,task_tag, time_limit, priority)

func submit_task_array_parameterized(instance: Object, method: String, parameter: Array,task_tag : String, time_limit : float = task_time_limit, priority:int = default_priority):
	return pool.submit_task_array_parameterized(instance, method, parameter ,task_tag, time_limit, priority)

func submit_task_as_only_parameter(instance: Object, method: String ,task_tag : String, time_limit : float = task_time_limit, priority:int = default_priority):
	return pool.submit_task_as_only_parameter(instance, method,task_tag, time_limit , priority)

func submit_task_unparameterized_if_no_parameter(instance: Object, method: String, task_tag : String,parameter = null, time_limit : float = task_time_limit, priority:int = default_priority):
	return pool.submit_task_unparameterized_if_no_parameter(instance, method, parameter ,task_tag, time_limit, priority)

