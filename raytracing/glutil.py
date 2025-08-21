import contextlib, sys
from OpenGL import GL as gl
import glfw
import logging
import ctypes


# Create a glfw window with given window width, height, and title.
@contextlib.contextmanager
def create_main_window(width, height, title=__name__):
    if not glfw.init():
        sys.exit(1)
    try:
        # Set up window configuration.
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        # Create a window.
        window = glfw.create_window(width, height, title, None, None)
        if not window:
            sys.exit(1)

        # Make the window as the gl content context.
        glfw.make_context_current(window)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        yield window
    finally:
        glfw.terminate()


# Create and bind a vertex array object.
@contextlib.contextmanager
def create_vertex_array_object():
    vertex_array_id = gl.glGenVertexArrays(1)
    try:
        gl.glBindVertexArray(vertex_array_id)
        yield
    finally:
        gl.glBindVertexArray(0)
        gl.glDeleteVertexArrays(1, [vertex_array_id])


# Create and bind a vertex buffer object.
@contextlib.contextmanager
def create_vertex_buffer_object(vertex_data):
    vertex_buffer_id = gl.glGenBuffers(1)
    vertex_attribute_id = 0
    try:
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vertex_buffer_id)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            vertex_data.nbytes,
            vertex_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            gl.GL_STATIC_DRAW,
        )
        gl.glVertexAttribPointer(vertex_attribute_id, 3, gl.GL_FLOAT, False, 0, None)
        gl.glEnableVertexAttribArray(vertex_attribute_id)
        yield
    finally:
        gl.glDisableVertexAttribArray(vertex_attribute_id)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glDeleteBuffers(1, [vertex_buffer_id])


@contextlib.contextmanager
def create_index_buffer_object(index_data):
    index_buffer_id = gl.glGenBuffers(1)
    try:
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, index_buffer_id)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER,
            index_data.nbytes,
            index_data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            gl.GL_STATIC_DRAW,
        )
        yield
    finally:
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        gl.glDeleteBuffers(1, [index_buffer_id])


# Load vertex and fragment shaders and attached them to a program.
@contextlib.contextmanager
def load_shaders(vertex_shader_path, fragment_shader_path):
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.ERROR)

    program_id = gl.glCreateProgram()
    attached_shaders = []
    try:
        # Load and attach the vertex and fragment shaders.
        for shader_type, shader_path in zip(
            [gl.GL_VERTEX_SHADER, gl.GL_FRAGMENT_SHADER],
            [vertex_shader_path, fragment_shader_path],
        ):
            with open(shader_path, "r") as shader_file:
                shader_id = gl.glCreateShader(shader_type)
                gl.glShaderSource(shader_id, shader_file.read())
                gl.glCompileShader(shader_id)
                success = gl.glGetShaderiv(shader_id, gl.GL_COMPILE_STATUS)
                if not success:
                    msg = gl.glGetShaderInfoLog(shader_id)
                    log.error(msg)
                    sys.exit(1)
                gl.glAttachShader(program_id, shader_id)
                attached_shaders.append(shader_id)

        # Link the program.
        gl.glLinkProgram(program_id)
        success = gl.glGetProgramiv(program_id, gl.GL_LINK_STATUS)
        if not success:
            msg = gl.glGetProgramInfoLog(program_id)
            log.error(msg)
            sys.exit(1)

        gl.glUseProgram(program_id)
        yield program_id
    finally:
        for shader_id in attached_shaders:
            gl.glDetachShader(program_id, shader_id)
            gl.glDeleteShader(shader_id)
        gl.glUseProgram(0)
        gl.glDeleteProgram(program_id)


def run_render_loop(window, render_func):
    while not glfw.window_should_close(window):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        render_func()
        glfw.swap_buffers(window)
        glfw.poll_events()
    glfw.destroy_window(window)
