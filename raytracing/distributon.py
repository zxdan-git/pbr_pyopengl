import numpy as np
from numpy.typing import NDArray

from .util import search_interval


class Distribution1D:
    def __init__(self, values):
        self.values_num = len(values)
        self.cdfs = np.zeros(self.values_num + 1, dtype=np.float32)
        for i in range(1, self.values_num + 1):
            self.cdfs[i] = self.cdfs[i - 1] + values[i - 1] / self.values_num

        self.values_int = self.cdfs[-1]
        for i in range(len(self.cdfs)):
            self.cdfs[i] /= self.values_int

    def sample(self, u: np.float32):
        sample_idx = search_interval(self.cdfs, u, 0, self.values_num)
        if sample_idx < 0 or sample_idx >= self.values_num:
            return np.random.randint(0, self.values_num)
        return sample_idx


class Distribution2D:
    def __init__(self, values_list):
        self.distributions = [Distribution1D(values) for values in values_list]
        self.values_distribution = Distribution1D(
            [distribution.values_int for distribution in self.distributions]
        )

    def sample(self, u: NDArray[np.float32]):
        sample_idx_1 = self.values_distribution.sample(u[0])
        sample_idx_2 = self.distributions[sample_idx_1].sample(u[1])
        return (sample_idx_1, sample_idx_2)
