import numpy as np

EPSILON = 1e-3
INF = np.float32(np.inf)
ZERO3F = np.zeros(3, dtype=np.float32)
ZERO3F.setflags(write=False)


def zero3f():
    return np.zeros(3, dtype=np.float32)


def zero2f():
    return np.zeros(2, dtype=np.float32)


def one3f():
    return np.ones(3, dtype=np.float32)


def red():
    return np.array([1, 0, 0], dtype=np.float32)


def green():
    return np.array([0, 1, 0], dtype=np.float32)


def blue():
    return np.array([0, 0, 1], dtype=np.float32)


def white():
    return np.array([1, 1, 1], dtype=np.float32)
