from OpenGL import GL as gl
import glfw
import raytracing.glutil as glutil
import raytracing.transform as transform
from raytracing.shape import Shape, Sphere, Cube, Triangle
from raytracing.ray import Ray
import raytracing.util as util
import numpy as np

rotate_angle = 0.01
rotate_axis = util.normalize(np.array([1, 1, 1], dtype = np.float32))

camera_pos = np.array([0, 0, 5], dtype = np.float32)
camera_target = np.array([0, 0, 0], dtype = np.float32)
camera_up = np.array([0, 1, 0], dtype = np.float32)

fov = 60
near = 1.0
far = 1000.0

shape_idx = 0

def setup_scene(window, program_id):
    view_mat_loc = gl.glGetUniformLocation(program_id, "view")
    view_mat = transform.world_to_camera(
            camera_pos,
            camera_target,
            camera_up
        )
    gl.glUniformMatrix4fv(view_mat_loc, 1, True, view_mat)

    projection_mat_loc = gl.glGetUniformLocation(program_id, "projection")
    window_width, window_height = glfw.get_window_size(window)
    projection_mat = transform.perspective(
            fov,
            float(window_width) / window_height,
            near,
            far
        )
    gl.glUniformMatrix4fv(projection_mat_loc, 1, True, projection_mat)

def ray_shape_intersect_test(window, program_id, shapes):
    global shape_idx
    n_shapes = len(shapes)
    if not 0 <= shape_idx < n_shapes:
        shape_idx = (shape_idx + n_shapes) % n_shapes

    shape = shapes[shape_idx]
    model_mat_loc = gl.glGetUniformLocation(program_id, "model")
    model_mat = shape.transform() @ transform.rotate(
            rotate_axis,
            rotate_angle
        )
    shape.set_transform(model_mat)

    window_width, window_height = glfw.get_window_size(window)
    inv_aspect = float(window_height) / window_width
    mouse_x, mouse_y = glfw.get_cursor_pos(window)
    mouse_in_camera = np.array([
        float(mouse_x) / window_width * 2 - 1,
        (float(window_height - mouse_y) / window_height * 2 - 1) * inv_aspect,
        -1 / np.tan(fov * 0.5 * np.pi / 180),
        1], dtype = np.float32)
    mouse_in_world = transform.camera_to_world(
            camera_pos,
            camera_target,
            camera_up
        ) @ mouse_in_camera
    mouse_in_world = mouse_in_world[:3]
    
    ray = Ray(camera_pos, util.normalize(mouse_in_world - camera_pos))
    intersect_t = shape.ray_intersect(ray)

    mode_loc = gl.glGetUniformLocation(program_id, "mode")

    index_offset = np.sum(
            [shapes[i].vertex().shape[0] for i in range(shape_idx)]
        , dtype = np.uint32)
    
    if shape.paint_mode() & Shape.PaintMode.FACE:
        with glutil.create_index_buffer_object(
                shape.face_index() + index_offset):
            gl.glUniformMatrix4fv(
                    model_mat_loc,
                    1,
                    True,
                    shape.transform()
                )
            if intersect_t is None:
                gl.glUniform1i(mode_loc, 0)
            else:
                gl.glUniform1i(mode_loc, 2)
            gl.glDrawElements(
                    gl.GL_TRIANGLES,
                    len(shape.face_index()),
                    gl.GL_UNSIGNED_INT,
                    None
                )
    
    if shape.paint_mode() & Shape.PaintMode.LINE:
        with glutil.create_index_buffer_object(
                shape.line_index() + index_offset):
            gl.glUniformMatrix4fv(
                    model_mat_loc,
                    1,
                    True,
                    shape.transform() @ transform.scale(1.01, 1.01, 1.01)
                )
            gl.glUniform1i(mode_loc, 1)
            gl.glDrawElements(
                    gl.GL_LINES,
                    len(shape.line_index()),
                    gl.GL_UNSIGNED_INT,
                    None
                )
            
def key_callback(window, key, scancode, action, mods):
    global shape_idx
    if action == glfw.PRESS:
        if key == glfw.KEY_DOWN:
            shape_idx += 1
        elif key == glfw.KEY_UP:
            shape_idx -= 1

if __name__ == "__main__":
    shapes = []
    sphere = Sphere(20, 40)
    sphere.set_paint_mode(Sphere.PaintMode.FACE_AND_LINE)
    shapes.append(sphere)

    cube = Cube()
    cube.set_paint_mode(Sphere.PaintMode.FACE_AND_LINE)
    shapes.append(cube)

    triangle = Triangle(
        np.array([0, 0.5, 0], dtype = np.float32),
        np.array([-0.5, -0.5, 0], dtype = np.float32),
        np.array([0.5, -0.5, 0], dtype = np.float32),
        )
    triangle.set_paint_mode(Sphere.PaintMode.FACE_AND_LINE)
    shapes.append(triangle)

    vertex = np.concatenate([shape.vertex() for shape in shapes], axis = 0)
    with glutil.create_main_window(1024, 768) as window:
        glfw.set_key_callback(window, key_callback)
        with glutil.load_shaders(
            "shaders/ray_shape_intersect.vertex",
            "shaders/ray_shape_intersect.fragment") as program_id:
            with glutil.create_vertex_array_object():
                with glutil.create_vertex_buffer_object(vertex.flatten()):
                        setup_scene(window, program_id)
                        glutil.run_render_loop(
                            window,
                            lambda: ray_shape_intersect_test(
                                window,
                                program_id,
                                shapes
                            )
                        )