import numpy as np

class Interval:
    def __init__(self, lower = np.inf, upper = -np.inf):
        self.lower = lower
        self.upper = upper
    
    @staticmethod
    def union(inv_1, inv_2):
        return Interval(
            lower = np.min([inv_1.lower, inv_2.lower]),
            upper = np.max([inv_1.upper, inv_2.upper]))
    
    @staticmethod
    def intersect(inv_1, inv_2):
        return Interval(
            lower = np.max([inv_1.lower, inv_2.lower]),
            upper = np.min([inv_1.upper, inv_2.upper]))
    
    def empty(self):
        return self.lower > self.upper
    
    def to_array(self):
        return np.array([self.lower, self.upper])