tool
extends Spatial

export var target_mesh_instance_path : NodePath = "MeshInstance"
onready var target_mesh_instance = get_node(target_mesh_instance_path)
export var static_body_instance_path : NodePath = "StaticBody"
onready var static_body_instance = get_node(static_body_instance_path)
export var collision_instance_path : NodePath = "StaticBody/CollisionShape"
onready var collision_instance = get_node(collision_instance_path)
export var radius : float = 5
export var height : float = 2
export var angle_range : int = 180
export var n_segments : int = 16

func _ready():
	var bottom = -height * 0.5
	var top = height * 0.5
	var angle_range_rad = deg2rad(angle_range)
	var angle_step_size = angle_range_rad / n_segments
	var angles = []
	for i in range(n_segments+1):
		angles.append(i * angle_step_size - 0.5 * angle_range_rad)
	
	var st = SurfaceTool.new()
	st.begin(Mesh.PRIMITIVE_TRIANGLES)
	var material = SpatialMaterial.new()
	material.flags_transparent = true
	material.flags_unshaded = true
	st.set_material(material)
	var x_left
	var x_right
	var z_left
	var z_right
	var pos_tl
	var pos_tr
	var pos_bl
	var pos_br
	var uv_l
	var uv_r
	for i in range(n_segments):
		# Create quads. 
		# Our perspektive is that we look from +z to -z. So the mesh has negative z coordinates.
		x_left = radius * sin(angles[i])
		x_right = radius * sin(angles[i + 1])
		z_left = -radius * cos(angles[i])
		z_right = -radius * cos(angles[i + 1])
		pos_tl = Vector3(x_left, top, z_left)
		pos_tr = Vector3(x_right, top, z_right)
		pos_bl = Vector3(x_left, bottom, z_left)
		pos_br = Vector3(x_right, bottom, z_right)
		uv_l = float(i) / float(n_segments)
		uv_r = float(i + 1) / float(n_segments)
		add_quad(st, pos_tl, pos_tr, pos_bl, pos_br, uv_l, uv_r)
	
	st.index()
	# Commit to a mesh.
	var mesh = st.commit()
	
	
	target_mesh_instance.mesh = mesh
	
	collision_instance.shape = mesh.create_trimesh_shape()
	static_body_instance.screen_angle_range = angle_range
	
	
func add_quad(st : SurfaceTool, pos_tl : Vector3, pos_tr : Vector3, pos_bl : Vector3, pos_br : Vector3, uv_l : float, uv_r : float):
	# Surface tool requires clockwise winding order.
	# For some reason, Godot expects that we look from positive z to negative z.
	
	var uv_t = 0
	var uv_b = 1
	
	# first triangle
	st.add_normal(-pos_tl)
	st.add_uv(Vector2(uv_l, uv_t))
	st.add_vertex(pos_tl)

	st.add_normal(-pos_tr)
	st.add_uv(Vector2(uv_r, uv_t))
	st.add_vertex(pos_tr)
	
	st.add_normal(-pos_bl)
	st.add_uv(Vector2(uv_l, uv_b))
	st.add_vertex(pos_bl)
	
	# second triangle
	st.add_normal(-pos_tr)
	st.add_uv(Vector2(uv_r, uv_t))
	st.add_vertex(pos_tr)
	
	st.add_normal(-pos_br)
	st.add_uv(Vector2(uv_r, uv_b))
	st.add_vertex(pos_br)

	st.add_normal(-pos_bl)
	st.add_uv(Vector2(uv_l, uv_b))
	st.add_vertex(pos_bl)

