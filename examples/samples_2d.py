import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from raytracing.util import (
    uniform_sample_hemisphere,
    uniform_sample_sphere,
    uniform_sample_disk,
    concentric_sample_disk,
    cosine_sample_hemisphere,
    uniform_sample_direction_in_cone,
    uniform_sample_triangle,
)


def navie_sample_disk(u: NDArray[np.float32]) -> NDArray[np.float32]:
    r = u[0]
    theta = 2 * np.pi * u[1]
    return np.array([r * np.cos(theta), r * np.sin(theta), 0], dtype=np.float32)


def show_sample_plot(ax, title, sample_func, *args, **kwargs):
    points = np.array(
        [sample_func(np.random.rand(2), *args, **kwargs) for _ in range(1000)]
    )
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5)
    ax.set_title(title)

    x_max = np.max(np.abs(points[:, 0])) + 0.1
    ax.set_xlim(-x_max, x_max)
    y_max = np.max(np.abs(points[:, 1])) + 0.1
    ax.set_ylim(-y_max, y_max)
    z_min, z_max = np.min(points[:, 2]) - 0.1, np.max(points[:, 2]) + 0.1
    ax.set_zlim(np.min([0, z_min]), z_max)

    ax.set_aspect("equal", "box")


if __name__ == "__main__":
    _, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={"projection": "3d"})
    show_sample_plot(
        axes[0], "Uniformly sample a hemisphere", uniform_sample_hemisphere
    )
    show_sample_plot(
        axes[1], "Consine weighted sample a hemisphere", cosine_sample_hemisphere
    )

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    show_sample_plot(ax, "Uniformly sample a sphere", uniform_sample_sphere)

    _, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={"projection": "3d"})
    show_sample_plot(axes[0], "Naviely sample a disk", navie_sample_disk)
    show_sample_plot(axes[1], "Uniformly sample a disk", uniform_sample_disk)
    show_sample_plot(axes[2], "Concentricly sample a disk", concentric_sample_disk)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    show_sample_plot(
        ax,
        "Uniformly sample a cone",
        uniform_sample_direction_in_cone,
        theta_max=np.pi / 4,
    )

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    show_sample_plot(
        ax,
        "Uniformly sample a triangle",
        uniform_sample_triangle,
        vertices=[
            np.array([-1, 0, 0], dtype=np.float32),
            np.array([1, 0, 0], dtype=np.float32),
            np.array([0, 0, 1], dtype=np.float32),
        ],
    )
    plt.show()
