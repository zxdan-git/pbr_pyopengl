import imageio
import numpy as np
from numpy.typing import NDArray
from os import path

from .transform import camera_to_world
from .util import normalize
from .ray import Ray


class Camera:
    def __init__(
        self,
        pos=np.zeros(3, dtype=np.float32),
        look_at=np.array([0, 0, -1], dtype=np.float32),
        up=np.array([0, 1, 0], dtype=np.float32),
        fov=60,
        film_width=256,
        film_height=256,
    ):
        self.__pos = pos
        self.__look_at = look_at
        self.__up = up
        self.__fov = np.float32(fov)
        self.__film_width = np.uint32(film_width)
        self.__film_height = np.uint32(film_height)
        self.__update_camera_parameters()
        self.__rng = np.random.default_rng()

    @property
    def pos(self):
        return self.__pos.copy()

    @pos.setter
    def pos(self, new_pos):
        self.__pos = new_pos
        self.__update_camera_parameters()

    @property
    def look_at(self):
        return self.__look_at.copy()

    @look_at.setter
    def look_at(self, new_look_at):
        self.__look_at = new_look_at
        self.__update_camera_parameters()

    @property
    def up(self):
        return self.__pos.copy()

    @up.setter
    def up(self, new_up):
        self.__up = new_up
        self.__update_camera_parameters()

    @property
    def fov(self):
        return self.__fov.copy()

    @fov.setter
    def fov(self, new_fov):
        self.__fov = new_fov
        self.__update_camera_parameters()

    @property
    def film_width(self):
        return self.__film_width.copy()

    @film_width.setter
    def film_width(self, new_film_width):
        self.__film_width = new_film_width
        self.__reset_film()

    @property
    def film_height(self):
        return self.__film_height.copy()

    @film_height.setter
    def film_height(self, new_film_height):
        self.__film_height = new_film_height
        self.__reset_film()

    @property
    def film_size(self):
        return np.array([self.__film_width, self.__film_height], dtype=np.uint32)

    @film_size.setter
    def film_size(self, new_size: NDArray[np.uint32]):
        self.film_width = new_size[0]
        self.film_height = new_size[1]
        self.__reset_film()

    def generate_view_ray_from(self, row, col):
        i_row = np.clip(row, 0, self.__film_height - 1)
        i_col = np.clip(col, 0, self.__film_width - 1)
        x_in_cam = np.float32((i_col + self.__rng.random()) / self.__film_width) * 2 - 1
        y_in_cam = 1 - 2 * np.float32(
            (i_row + self.__rng.random()) / self.__film_height
        )
        target = self.__camera_to_world @ np.array(
            [x_in_cam, y_in_cam, self.__f, 1], dtype=np.float32
        )
        return Ray(self.__pos, normalize(target[:3] - self.__pos))

    def write_to(self, row, col, rgb: NDArray[np.float32]):
        self.__film[row][col] = rgb

    def save_film(self, output_path, file_name):
        imageio.imwrite(path.join(output_path, file_name + ".tiff"), self.__film)

    def __update_camera_parameters(self):
        self.__camera_to_world = camera_to_world(self.__pos, self.__look_at, self.__up)
        self.__f = 1.0 / -np.tan(0.5 * self.__fov * np.pi / 180)
        self.__reset_film()

    def __reset_film(self):
        self.__film = np.zeros(
            [self.__film_height, self.film_width, 3], dtype=np.float32
        )
