import numpy as np

from .interval import Interval
from .ray import Ray


class AABB:
    def __init__(
        self,
        x_min=np.inf,
        x_max=-np.inf,
        y_min=np.inf,
        y_max=-np.inf,
        z_min=np.inf,
        z_max=-np.inf,
    ):
        self.__inv_x = Interval(x_min, x_max)
        self.__inv_y = Interval(y_min, y_max)
        self.__inv_z = Interval(z_min, z_max)

    @staticmethod
    def union(bbx_1, bbx_2):
        inv_x = Interval.union(bbx_1.range_x(), bbx_2.range_x())
        inv_y = Interval.union(bbx_1.range_y(), bbx_2.range_y())
        inv_z = Interval.union(bbx_1.range_z(), bbx_2.range_z())
        return AABB(
            inv_x.lower, inv_x.upper, inv_y.lower, inv_y.upper, inv_z.lower, inv_z.upper
        )

    @staticmethod
    def intersect(bbx_1, bbx_2):
        inv_x = Interval.intersect(bbx_1.range_x(), bbx_2.range_x())
        inv_y = Interval.intersect(bbx_1.range_y(), bbx_2.range_y())
        inv_z = Interval.intersect(bbx_1.range_z(), bbx_2.range_z())
        return AABB(
            inv_x.lower, inv_x.upper, inv_y.lower, inv_y.upper, inv_z.lower, inv_z.upper
        )

    def embrace(self, pos):
        self.__inv_x = Interval.union(self.range_x(), Interval(pos[0], pos[0]))
        self.__inv_y = Interval.union(self.range_y(), Interval(pos[1], pos[1]))
        self.__inv_z = Interval.union(self.range_z(), Interval(pos[2], pos[2]))

    def center(self):
        if self.empty():
            raise ValueError("Empty bounding box has no center.")
        return np.array(
            [
                self.__inv_x.to_array().mean(),
                self.__inv_y.to_array().mean(),
                self.__inv_z.to_array().mean(),
            ],
            dtype=np.float32,
        )

    def range_x(self):
        return self.__inv_x

    def range_y(self):
        return self.__inv_y

    def range_z(self):
        return self.__inv_z

    def get_range(self, idx):
        if idx == 0:
            return self.__inv_x
        elif idx == 1:
            return self.__inv_y
        elif idx == 2:
            return self.__inv_z
        raise IndexError("index out of range")

    def get_max_axis(self):
        return np.argmax([self.get_range(i).size() for i in range(3)])

    def empty(self):
        return self.__inv_x.empty() or self.__inv_y.empty() or self.__inv_z.empty()

    def surface_area(self):
        len_x = self.__inv_x.size()
        len_y = self.__inv_y.size()
        len_z = self.__inv_z.size()

        return 2 * (len_x * len_y + len_x * len_z + len_y * len_z)

    def offset(self, pos):
        return np.array(
            [
                (pos[0] - self.__inv_x.lower) / self.__inv_x.size(),
                (pos[1] - self.__inv_y.lower) / self.__inv_y.size(),
                (pos[2] - self.__inv_z.lower) / self.__inv_z.size(),
            ],
            dtype=np.float32,
        )

    def ray_intersect(self, ray: Ray):
        time_inv = Interval(-1, ray.t_max)
        for i in range(3):
            inv_i = self.get_range(i).to_array()
            if np.isclose(ray.dir[i], 0):
                continue
            time_range = (inv_i - ray.pos[i]) / np.float32(ray.dir[i])
            time_inv = Interval.intersect(
                time_inv, Interval(np.min(time_range), np.max(time_range))
            )
            if time_inv.empty():
                return None
        if time_inv.lower >= 0:
            return time_inv.lower
        elif time_inv.upper < ray.t_max:
            return time_inv.upper
        return None
