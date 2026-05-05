import numpy as np

from .typing import Vec2f, vec2f


class Interval:
    def __init__(self, lower=np.inf, upper=-np.inf):
        self.lower = lower
        self.upper = upper

    @staticmethod
    def union(inv_1, inv_2):
        return Interval(
            lower=np.min([inv_1.lower, inv_2.lower]),
            upper=np.max([inv_1.upper, inv_2.upper]),
        )

    @staticmethod
    def intersect(inv_1, inv_2):
        return Interval(
            lower=np.max([inv_1.lower, inv_2.lower]),
            upper=np.min([inv_1.upper, inv_2.upper]),
        )

    def size(self):
        if self.empty():
            return 0
        return self.upper - self.lower

    def empty(self):
        return self.lower > self.upper

    def to_array(self) -> Vec2f:
        return vec2f(self.lower, self.upper)
