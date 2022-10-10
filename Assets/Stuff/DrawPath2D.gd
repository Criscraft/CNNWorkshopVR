extends Path2D

func _ready():
	$Line2D.points = curve.tessellate(5, 4)
