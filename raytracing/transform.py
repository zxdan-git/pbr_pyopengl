import numpy as np
from raytracing.util import normalize

def world_to_camera(pos, look_at, up):
    '''
    Transform a position p in the world space to the the camera space.

    The transformed coordinates p' is to project vector p - pos on camera basis.

    p'x = camera_x @ (p - pos) = camera_x @ p - camera_x @ pos
        = [camera_x, -camera_x @ pos] @ [p, 1]

    p'y = camera_y @ (p - pos) = camera_y @ p - camera_y @ pos
        = [camera_y, -camera_y @ pos] @ [p, 1]

    p'z = camera_z @ (p - pos) = camera_z @ p - camera_z @ pos
        = [camera_z, -camera_z @ pos] @ [p, 1]

    Therefore,
    M = [[camera_x, -camera_x @ pos],
         [camera_y, -camera_y @ pos],
         [camera_z, -camera_z @ pos]]
    '''
    view_mat = np.zeros((4, 4), dtype = np.float32)
    camera_z = normalize(pos -look_at)
    camera_x = normalize(np.cross(normalize(up), camera_z))
    camera_y = normalize(np.cross(camera_z, camera_x))
    view_mat[0] = np.append(camera_x, -np.dot(camera_x, pos))
    view_mat[1] = np.append(camera_y, -np.dot(camera_y, pos))
    view_mat[2] = np.append(camera_z, -np.dot(camera_z, pos))
    view_mat[3] = np.array([0, 0, 0, 1], dtype = np.float32)
    return view_mat

def camera_to_world(pos, look_at, up):
    '''
    Transform a postion p in the camera space to the world space.

    The camera basis (1, 0, 0, 0), (0, 1, 0, 0), and (0, 0, 1, 0) will be
    transformed to camera_x, camera_y, and camera_z in world space.

    M @ [1, 0, 0, 0] = M[:, 0] = [camera_x, 0]

    M @ [0, 1, 0, 0] = M[:, 1] = [camera_y, 0]

    M @ [0, 0, 1, 0] = M[:, 2] = [camera_z, 0]

    And the position (0, 0, 0, 1) will be mapped to camera position in world
    space.

    M @ [0, 0, 0, 1] = M[:, 3] = [pos, 1]

    Therefore,

    M = [[camera_x, 0]^T [camera_y, 0]^T [camera_z, 0]^T [pos, 1]^T]
    '''
    view_mat = np.zeros((4, 4), dtype = np.float32)
    view_z = normalize(pos -look_at)
    view_x = normalize(np.cross(normalize(up), view_z))
    view_y = normalize(np.cross(view_z, view_x))
    view_mat[:3, 0] = view_x
    view_mat[:3, 1] = view_y
    view_mat[:3, 2] = view_z
    view_mat[:, 3] = np.append(pos, 1)
    return view_mat

def perspective(fov, aspect, near, far):
    '''
    Transform a position p in camera space to the clip space.

    fov is field of view on XZ plane.

    aspect is frame width / height.

    near to far is the range of the rendering depth.

    The each coordinate is transformed in a visible range of [-1, 1].

    The focal length f is 1 / tan(fov / 2). According to the similar ratio:
    
    x' = x *  f / -z

    Note that we are at camera origin (0, 0, 0) and look backward from camera_z,
    so all visible position's z value should be negative. That's why we use -z
    here.
    
    Smiliarly,

    y' = y * f / -z

    but this will map the vertical visible points in range [-1/aspect,
    1/aspect], so we should scale it with aspect. Therefore,

    y' = y * aspect * f / -z

    For ease of depth testing, we should preserve the depth relationship in z'.
    We know when z = -near, z' = -1, when z = -far, z' = 1, and the change rate
    of z' should be aligned with x' and y', aka. (alpha / -z). Suppose

    z' = (alpha . z + beta) / -z

    By substituting (-near, -1) and (-far, 1) into the formula, we can get

    alpha = -(far + near) / (far - near)
    
    beta = -2 * far * near / (far  - near)
    '''
    f = 1 / np.tan(0.5 * fov * np.pi / 180)
    alpha = -(far + near) / (far - near)
    beta = -2 * far * near / (far - near)
    return np.array(
        [
            [f, 0, 0, 0],
            [0, f * aspect, 0, 0],
            [0, 0, alpha, beta],
            [0, 0, -1, 0]
        ],
        dtype = np.float32)

def scale(sx, sy, sz):
    return np.array(
        [
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1]
        ],
        dtype = np.float32)

def rotate_X(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [1, 0, 0, 0],
            [0, c, -s, 0],
            [0, s, c, 0],
            [0, 0, 0, 1]
        ],
        dtype = np.float32)

def rotate_Y(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [c, 0, s, 0],
            [0, 1, 0, 0],
            [-s, 0, c, 0],
            [0, 0, 0, 1]
        ],
        dtype = np.float32)

def rotate_Z(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array(
        [
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ],
        dtype = np.float32)

def rotate(axis, angle):
    '''
    Rotate any vector p around axis by angle.
    Decompose the vector p into two components:

    1. component parallel to axis: p.a.a
    2. component perpendicular to axis: p - p.a.a

    The parallel component will not be changed during rotation but only the
    perpendicular part. On the plane perpendicular to the axis, the
    rotated perpendicular component can then be decomposed into two parts:
    
    1. (p - p.a.a) * cos(angle)
    2. (a x p) * sin(angle) (Note the order)
    
    p' = p.a.a + (p - p.a.a) * cos(angle) + (a x p) * sin(angle)
        = p.c + (1 - c)p.a.a + (a x p).s
    '''
    axis = normalize(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    ax, ay, az = axis[0], axis[1], axis[2]
    return np.array(
        [
            [(1 - c) * ax * ax + c, (1 - c) * ax * ay - az * s, (1 - c) * ax * az + ay * s, 0],
            [(1 - c) * ay * ax + az * s, (1 - c) * ay * ay + c, (1 - c) * ay * az - ax * s, 0],
            [(1 - c) * az * ax - ay * s, (1 - c) * az * ay + ax * s, (1 - c) * az * az + c, 0],
            [0, 0, 0, 1]
        ],
        dtype = np.float32)

