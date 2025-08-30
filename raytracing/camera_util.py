from .transform import camera_to_world
from .ray import Ray
from .util import normalize
import numpy as np


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
        camera_to_world(camera_pos, camera_center, camera_up) @ target_in_camera
    )
    return Ray(camera_pos, normalize(target_in_world[:3] - camera_pos))
