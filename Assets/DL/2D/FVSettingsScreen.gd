extends PanelContainer

var fv_settings_resource : FVSettingsResource setget set_fv_settings_resource
var settings_changed : bool = false
onready var debouncing_timer = $DebouncingTimer
onready var option_button_mode = $VBoxContainer/OptionButtonMode
onready var slider_epochs : HSlider = $VBoxContainer/SliderEpochs
onready var value_epochs : Label = $VBoxContainer/ValueEpochs
onready var slider_learning_rate : HSlider = $VBoxContainer/SliderLearningRate
onready var value_learning_rate : Label = $VBoxContainer/ValueLearningRate
onready var slider_rotation : HSlider = $VBoxContainer/SliderRotation
onready var value_rotation : Label = $VBoxContainer/ValueRotation
onready var slider_blur_sigma : HSlider = $VBoxContainer/SliderBlurSigma
onready var value_blur_sigma : Label = $VBoxContainer/ValueBlurSigma
onready var slider_roll : HSlider = $VBoxContainer/SliderRoll
onready var value_roll : Label = $VBoxContainer/ValueRoll
onready var slider_fraction_to_maximize : HSlider = $VBoxContainer/SliderFractionToMaximize
onready var value_fraction_to_maximize : Label = $VBoxContainer/ValueFractionToMaximize

func _ready():
	option_button_mode.add_item("Average")
	option_button_mode.add_item("Center pixel")
	option_button_mode.add_item("Fraction to maximize")


func set_fv_settings_resource(fv_settings_resource_):
	fv_settings_resource = fv_settings_resource_
	option_button_mode.select(fv_settings_resource.mode)
	slider_epochs.value = fv_settings_resource.epochs
	value_epochs.text = str(fv_settings_resource.epochs)
	slider_learning_rate.value = fv_settings_resource.lr
	value_learning_rate.text = str(fv_settings_resource.lr)
	slider_rotation.value = fv_settings_resource.degrees
	value_rotation.text = str(fv_settings_resource.degrees)
	slider_blur_sigma.value = fv_settings_resource.blur_sigma
	value_blur_sigma.text = str(fv_settings_resource.blur_sigma)
	slider_roll.value = fv_settings_resource.roll
	value_roll.text = str(fv_settings_resource.roll)
	slider_fraction_to_maximize.value = fv_settings_resource.fraction_to_maximize
	value_fraction_to_maximize.text = str(fv_settings_resource.fraction_to_maximize)
	settings_changed = false
	

func _on_OptionButtonMode_item_selected(index):
	fv_settings_resource.mode = index
	settings_changed = true
	debouncing_timer._on_trigger()
	print("option button selected")


func _on_SliderEpochs_value_changed(value):
	value_epochs.text = str(value)
	fv_settings_resource.epochs = value
	settings_changed = true
	debouncing_timer._on_trigger()
	print("epochs changed")
	# update text


func _on_SliderLearningRate_value_changed(value):
	value_learning_rate.text = str(value)
	fv_settings_resource.lr = value
	settings_changed = true
	debouncing_timer._on_trigger()


func _on_SliderRotation_value_changed(value):
	value_rotation.text = str(value)
	fv_settings_resource.degrees = value
	settings_changed = true
	debouncing_timer._on_trigger()


func _on_SliderBlurSigma_value_changed(value):
	value_blur_sigma.text = str(value)
	fv_settings_resource.blur_sigma = value
	settings_changed = true
	debouncing_timer._on_trigger()


func _on_SliderRoll_value_changed(value):
	value_roll.text = str(value)
	fv_settings_resource.roll = value
	settings_changed = true
	debouncing_timer._on_trigger()


func _on_SliderFractionToMaximize_value_changed(value):
	value_fraction_to_maximize.text = str(value)
	fv_settings_resource.fraction_to_minimize = value
	settings_changed = true
	debouncing_timer._on_trigger()


func _on_DebouncingTimer_timeout():
	# Send the new settings to the server.
	if settings_changed:
		DLManager.set_fv_settings(fv_settings_resource.get_dict())
		settings_changed = false
