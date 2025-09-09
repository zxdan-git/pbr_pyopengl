from OpenGL import GL as gl
import glfw
import raytracing.glutil as glutil
import raytracing.transform as transform
from raytracing.shape import Shape, Sphere, Cube
from raytracing.ray import Ray
from raytracing.bounding_box import AABB
from raytracing.bounding_volume_hierarchy import BVH
from raytracing.bvh_util.build_node import BuildNode
import numpy as np
from typing import List
import random

camera_pos = np.array([10, 10, 10], dtype=np.float32)
camera_target = np.array([0, 0, 0], dtype=np.float32)
camera_up = np.array([0, 1, 0], dtype=np.float32)

fov = 60
near = 1.0
far = 1000.0

shapes: List[Shape] = []

bvh_level = 0
bvh_types = [
    BVH.Type.MID_POINT,
    BVH.Type.EQUAL_COUNT,
    BVH.Type.SAH,
    BVH.Type.MORTON_CODE,
]
bvh_type_names = ["MID POINT", "EQUAL_COUNT", "SAH", "MORTON_CODE"]
bvh_type_id = 0
bvh: BVH = None
bvh_nodes: List[List[BuildNode]] = []
bvh_cost = 0


def setup_scene(window, program_id):
    view_mat_loc = gl.glGetUniformLocation(program_id, "view")
    view_mat = transform.world_to_camera(camera_pos, camera_target, camera_up)
    gl.glUniformMatrix4fv(view_mat_loc, 1, True, view_mat)

    projection_mat_loc = gl.glGetUniformLocation(program_id, "projection")
    window_width, window_height = glfw.get_window_size(window)
    projection_mat = transform.perspective(
        fov, float(window_width) / window_height, near, far
    )
    gl.glUniformMatrix4fv(projection_mat_loc, 1, True, projection_mat)


def paint_bounding_box(bbx: AABB, index_offset, model_mat_loc):
    bbx_shape = Cube()
    with glutil.create_index_buffer_object(bbx_shape.line_index + index_offset):
        pos = bbx.center()
        sx = 0.5 * bbx.range_x().size()
        sy = 0.5 * bbx.range_y().size()
        sz = 0.5 * bbx.range_z().size()
        gl.glUniformMatrix4fv(
            model_mat_loc,
            1,
            True,
            transform.translate(pos[0], pos[1], pos[2]) @ transform.scale(sx, sy, sz),
        )
        gl.glDrawElements(
            gl.GL_LINES, len(bbx_shape.line_index), gl.GL_UNSIGNED_INT, None
        )


def paint_bbx_in_bvh(index_offset, model_mat_loc):
    for node in bvh_nodes[bvh_level]:
        paint_bounding_box(node.bbx, index_offset, model_mat_loc)


def bounding_volume_hierarchy_test(program_id, shapes: List[Shape], text_renderer):
    glutil.show_text(
        text_renderer,
        "UP: last BVH type",
        "DOWN: next BVH type",
        "LEFT: last BVH level",
        "RIGHT: next BVH level",
        "R: rotate the camera",
        "",
        "Current BVH INFO:",
        "   Name: %s" % bvh_type_names[bvh_type_id],
        "   Current Level: %d" % bvh_level,
        "   Ray Intersect Cost: %f" % bvh_cost,
    )
    view_mat_loc = gl.glGetUniformLocation(program_id, "view")
    view_mat = transform.world_to_camera(camera_pos, camera_target, camera_up)
    gl.glUniformMatrix4fv(view_mat_loc, 1, True, view_mat)

    model_mat_loc = gl.glGetUniformLocation(program_id, "model")
    index_offset = 0

    for shape in shapes:
        with glutil.create_index_buffer_object(shape.face_index + index_offset):
            gl.glUniformMatrix4fv(model_mat_loc, 1, True, shape.transform)
            gl.glDrawElements(
                gl.GL_TRIANGLES, len(shape.face_index), gl.GL_UNSIGNED_INT, None
            )
        index_offset += shape.vertex.shape[0]

    paint_bbx_in_bvh(index_offset, model_mat_loc)


def generate_bvh():
    global bvh, bvh_level, bvh_nodes, bvh_cost
    bvh = BVH(bvh_types[bvh_type_id], shapes)
    bvh_level = 0
    bvh_nodes = [[bvh.root]]
    bvh_cost = bvh.ray_intersect_cost()
    print("Generated", bvh_type_names[bvh_type_id], "with cost", bvh_cost)
    while True:
        children: List[BuildNode] = []
        for parent in bvh_nodes[-1]:
            if parent.left:
                children.append(parent.left)
            if parent.right:
                children.append(parent.right)
            for object in parent.objects:
                if isinstance(object, BuildNode):
                    children.append(object)
        if len(children):
            bvh_nodes.append(children)
        else:
            break


def key_callback(window, key, scancode, action, mods):
    global camera_pos, bvh_level, bvh_type_id
    if action == glfw.REPEAT or action == glfw.PRESS:
        if key == glfw.KEY_R:
            dist = camera_pos - camera_target
            t_dist = transform.rotate_Y(0.1) @ np.append(dist, 0)
            camera_pos = camera_target + t_dist[:3]

        if key == glfw.KEY_DOWN:
            bvh_level += 1
            if bvh_level == len(bvh_nodes):
                bvh_level = 0

        if key == glfw.KEY_UP:
            bvh_level -= 1
            if bvh_level == -1:
                bvh_level = len(bvh_nodes) - 1

        n_types = len(bvh_types)
        if key == glfw.KEY_LEFT:
            bvh_type_id = (n_types + bvh_type_id - 1) % n_types
            generate_bvh()

        if key == glfw.KEY_RIGHT:
            bvh_type_id = (n_types + bvh_type_id + 1) % n_types
            generate_bvh()


if __name__ == "__main__":
    for _ in range(10):
        sphere = Sphere(10, 20)
        sphere.transform = transform.translate(
            random.random() * 5, random.random() * 5, random.random() * 5
        ) @ transform.scale(random.random(), random.random(), random.random())
        shapes.append(sphere)

    bbx = AABB()
    for shape in shapes:
        bbx = AABB.union(bbx, shape.bounding_box)
    camera_target = bbx.center()

    vertex = np.concatenate(
        [shape.vertex for shape in shapes] + [Cube().vertex], axis=0
    )

    generate_bvh()

    with glutil.create_main_window(1024, 768) as window:
        with glutil.create_text_renderer(window) as text_renderer:
            glfw.set_key_callback(window, key_callback)
            with glutil.load_shaders(
                "shaders/bounding_volume_hierarchy.vertex",
                "shaders/bounding_volume_hierarchy.fragment",
            ) as program_id:
                with glutil.create_vertex_array_object():
                    with glutil.create_vertex_buffer_object(vertex.flatten()):
                        setup_scene(window, program_id)
                        glutil.run_render_loop(
                            window,
                            lambda: bounding_volume_hierarchy_test(
                                program_id, shapes, text_renderer
                            ),
                        )
