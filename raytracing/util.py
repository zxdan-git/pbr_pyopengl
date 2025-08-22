import numpy as np
import raytracing.transform as transform
from raytracing.ray import Ray


def normalize(v):
    return v / np.linalg.norm(v)


def det3x3(mat3x3):
    return (
        mat3x3[0, 0] * mat3x3[1, 1] * mat3x3[2, 2]
        - mat3x3[0, 0] * mat3x3[1, 2] * mat3x3[2, 1]
        + mat3x3[0, 1] * mat3x3[1, 2] * mat3x3[2, 0]
        - mat3x3[0, 1] * mat3x3[1, 0] * mat3x3[2, 2]
        + mat3x3[0, 2] * mat3x3[1, 0] * mat3x3[2, 1]
        - mat3x3[0, 2] * mat3x3[1, 1] * mat3x3[2, 0]
    )


def camera_ray(
    target_x,
    target_y,
    camera_pos,
    camera_center,
    camera_up,
    fov,
    film_width,
    film_height,
) -> Ray:
    """
    1. Get target position on the film in camera space:
        x = 2 * target_x / film_width - 1,
        y = (2 * (film_height - target_y) / film_height - 1) / aspect,
        z = -focal_length
    2. Transform the position to world space
    """
    inv_aspect = float(film_height) / film_width
    target_in_camera = np.array(
        [
            float(target_x) / film_width * 2 - 1,
            (float(film_height - target_y) / film_height * 2 - 1) * inv_aspect,
            -1 / np.tan(fov * 0.5 * np.pi / 180),
            1,
        ],
        dtype=np.float32,
    )
    target_in_world = (
        transform.camera_to_world(camera_pos, camera_center, camera_up)
        @ target_in_camera
    )
    return Ray(camera_pos, normalize(target_in_world[:3] - camera_pos))


def partition(array, pivot, start, end, value_func=lambda value: value):
    """
    Reorder the array in range [start, end) into two parts and return a
    partition position mid. The values in reordered part in [start, mid)
    will be smaller than pivot, and the values in [mid, end) will be
    large than or equal to the pivot.
    """
    i, j = start, end
    while i < j:
        while i < j and value_func(array[i]) < pivot:
            i += 1
        if i == j:
            return i
        while i < j and value_func(array[j - 1]) >= pivot:
            j -= 1
        if i == j:
            return j
        array[i], array[j - 1] = array[j - 1], array[i]
    return i
