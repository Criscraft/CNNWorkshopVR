extends Path2D

func set_curve(curve_):
	curve = curve_
	update_drawing()
	
func set_color(color):
	$Line2D.default_color = color

func update_drawing():
	$Line2D.points = curve.tessellate(5, 4)
