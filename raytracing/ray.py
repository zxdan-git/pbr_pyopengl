import numpy as np

class Ray:
    def __init__(self, pos, dir):
        self.pos = pos
        self.dir = dir

    @staticmethod
    def transform(ray, transform_mat):
        pos = transform_mat @ np.append(ray.pos, 1)
        dir = transform_mat @ np.append(ray.dir, 0)
        return Ray(pos[:3], dir[:3])
    
    def at(self, t):
        return self.pos + self.dir * t