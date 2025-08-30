from OpenGL import GL as gl
import glfw
import raytracing.glutil as glutil
import raytracing.transform as transform
from raytracing.shape import Shape, Sphere, Cube, Triangle
from raytracing.ray import Ray
from raytracing.bounding_box import AABB
import raytracing.util as util
from raytracing.camera_util import camera_ray
import numpy as np
from typing import List

camera_pos = np.array([10, 10, 10], dtype=np.float32)
camera_target = np.array([0, 0, 0], dtype=np.float32)
camera_up = np.array([0, 1, 0], dtype=np.float32)

fov = 60
near = 1.0
far = 1000.0


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


def multiple_shapes_intersection_test(window, program_id, shapes: List[Shape]):
    window_width, window_height = glfw.get_window_size(window)
    mouse_x, mouse_y = glfw.get_cursor_pos(window)
    view_ray = camera_ray(
        mouse_x,
        mouse_y,
        camera_pos,
        camera_target,
        camera_up,
        fov,
        window_width,
        window_height,
    )

    intersect_to_XZ = view_ray.at(-view_ray.pos[1] / view_ray.dir[1])
    ray = Ray(np.zeros(3, dtype=np.float32), util.normalize(intersect_to_XZ))
    ray.t_max = np.linalg.norm(intersect_to_XZ)

    model_mat_loc = gl.glGetUniformLocation(program_id, "model")
    mode_loc = gl.glGetUniformLocation(program_id, "mode")
    index_offset = 0
    for shape in shapes:
        shape.ray_intersect(ray)

    for shape in shapes:
        with glutil.create_index_buffer_object(shape.face_index + index_offset):
            gl.glUniformMatrix4fv(model_mat_loc, 1, True, shape.transform)
            intersect = shape.ray_intersect(ray)
            if (not intersect is None) and np.isclose(intersect, ray.t_max):
                gl.glUniform1i(mode_loc, 2)
            else:
                gl.glUniform1i(mode_loc, 0)
            gl.glDrawElements(
                gl.GL_TRIANGLES, len(shape.face_index), gl.GL_UNSIGNED_INT, None
            )
        index_offset += shape.vertex.shape[0]

    ray_index = np.array([0, 1], dtype=np.uint32) + index_offset
    with glutil.create_index_buffer_object(ray_index):
        gl.glUniformMatrix4fv(
            model_mat_loc,
            1,
            True,
            transform.rotate_Y(np.atan2(intersect_to_XZ[0], intersect_to_XZ[2]))
            @ transform.scale(ray.t_max, ray.t_max, ray.t_max),
        )
        gl.glUniform1i(mode_loc, 0)
        gl.glDrawElements(gl.GL_LINES, 2, gl.GL_UNSIGNED_INT, None)


if __name__ == "__main__":
    shapes: List[Shape] = []
    sphere = Sphere(20, 40)
    sphere.transform = transform.translate(1.5, 0, 3)
    shapes.append(sphere)

    sphere = Sphere(20, 40)
    sphere.transform = transform.translate(-1.5, 0, 5)
    shapes.append(sphere)

    cube = Cube()
    cube.transform = transform.translate(2, 0, 7)
    shapes.append(cube)

    triangle = Triangle(
        np.array([0, 0.5, 0], dtype=np.float32),
        np.array([-0.5, -0.5, 0], dtype=np.float32),
        np.array([0.5, -0.5, 0], dtype=np.float32),
    )
    triangle.transform = transform.translate(0, 0, 9)
    shapes.append(triangle)

    bbx = AABB()
    for shape in shapes:
        bbx = AABB.union(bbx, shape.bounding_box)
    camera_target = bbx.center()

    ray_vertext = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.float32)

    vertex = np.concatenate([shape.vertex for shape in shapes] + [ray_vertext], axis=0)
    with glutil.create_main_window(1024, 768) as window:
        with glutil.load_shaders(
            "shaders/ray_shape_intersect.vertex", "shaders/ray_shape_intersect.fragment"
        ) as program_id:
            with glutil.create_vertex_array_object():
                with glutil.create_vertex_buffer_object(vertex.flatten()):
                    setup_scene(window, program_id)
                    glutil.run_render_loop(
                        window,
                        lambda: multiple_shapes_intersection_test(
                            window, program_id, shapes
                        ),
                    )
