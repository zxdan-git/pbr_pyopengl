from typing import Annotated

import numpy as np
from numpy.typing import NDArray

Vec2u = Annotated[NDArray[np.uint32], "shape: (2,)"]

Vec2f = Annotated[NDArray[np.float32], "shape: (2,)"]

Vec3f = Annotated[NDArray[np.float32], "shape: (3,)"]

Mat4f = Annotated[NDArray[np.float32], "shape: (4, 4)"]

Array_f = NDArray[np.float32]

Array_u = NDArray[np.uint32]


def vec2u(u, v) -> Vec2u:
    return np.array([u, v], dtype=np.uint32)


def vec2f(x, y) -> Vec2f:
    return np.array([x, y], dtype=np.float32)


def vec3f(x, y, z) -> Vec3f:
    return np.array([x, y, z], dtype=np.float32)


def array_f(data) -> NDArray[np.float32]:
    return np.array(data, dtype=np.float32)


def array_u(data) -> NDArray[np.uint32]:
    return np.array(data, dtype=np.uint32)


def zeros_f(shape) -> NDArray[np.float32]:
    return np.zeros(shape, dtype=np.float32)
