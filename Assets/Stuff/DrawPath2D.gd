extends Path2D

func set_curve(curve_):
	curve = curve_
	update_drawing()

func update_drawing():
	$Line2D.points = curve.tessellate(5, 4)
