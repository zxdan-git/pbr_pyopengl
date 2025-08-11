import numpy as np
from raytracing.interval import Interval
from raytracing.ray import Ray

class AABB:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max):
        self.__inv_x = Interval(x_min, x_max)
        self.__inv_y = Interval(y_min, y_max)
        self.__inv_z = Interval(z_min, z_max)
    
    @staticmethod
    def union(bbx_1, bbx_2):
        inv_x = Interval.union(bbx_1.range_x(), bbx_2.range_x())
        inv_y = Interval.union(bbx_1.range_y(), bbx_2.range_y())
        inv_z = Interval.union(bbx_1.range_z(), bbx_2.range_z())
        return AABB(
                inv_x.lower, inv_x.upper,
                inv_y.lower, inv_y.upper,
                inv_z.lower, inv_z.upper
            )
    
    @staticmethod
    def intersect(bbx_1, bbx_2):
        inv_x = Interval.intersect(bbx_1.range_x(), bbx_2.range_x())
        inv_y = Interval.intersect(bbx_1.range_y(), bbx_2.range_y())
        inv_z = Interval.intersect(bbx_1.range_z(), bbx_2.range_z())
        return AABB(
            inv_x.lower, inv_x.upper,
            inv_y.lower, inv_y.upper,
            inv_z.lower, inv_z.upper
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

    def empty(self):
        return not self.__inv_x.empty() and \
               not self.__inv_y.empty() and \
               not self.__inv_z.empty()
    
    def ray_intersect(self, ray):
        time_inv = Interval(-np.inf, np.inf)
        for i in range(3):
            inv_i = self.get_range(i).to_array()
            time_range = (inv_i - ray.pos[i]) / ray.dir[i]
            time_inv = Interval.intersect(
                    time_inv,
                    Interval(np.min(time_range), np.max(time_range))
                )
            if time_inv.empty():
                return None

        if 0 < time_inv.lower < np.inf:
            return time_inv.lower
        elif 0 < time_inv.upper < np.inf:
            return time_inv.upper
        return None

        
