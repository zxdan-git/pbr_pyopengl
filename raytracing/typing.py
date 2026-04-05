from typing import Annotated

import numpy as np
from numpy.typing import NDArray

Vec2u = Annotated[NDArray[np.uint32], "shape: (2,)"]

Vec2f = Annotated[NDArray[np.float32], "shape: (2,)"]

Vec3f = Annotated[NDArray[np.float32], "shape: (3,)"]

Mat4f = Annotated[NDArray[np.float32], "shape: (4, 4)"]
