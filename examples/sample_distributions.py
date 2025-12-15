import matplotlib.pyplot as plt
import numpy as np

from raytracing.distributon import Distribution1D, Distribution2D


def sample_1d():
    values_num = 10
    values = np.random.rand(values_num)
    distribution_1d = Distribution1D(values)
    samples = [distribution_1d.sample(np.random.rand()) for _ in range(1000)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Distribution 1D")
    x = np.arange(values_num)
    axes[0].bar(x, values / values_num / distribution_1d.values_int)
    axes[0].set_xticks(x, [f"{i}" for i in x])
    axes[0].set_title("PDF")

    bins = np.arange(values_num + 1) - 0.5
    axes[1].hist(samples, bins=bins, width=0.8)
    axes[1].set_title("Sample Distribution")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{i}" for i in x])


def sample_2d():
    values_num = 10
    values = np.array([np.random.rand(values_num) for _ in range(values_num)])
    distribution_2d = Distribution2D(values)
    samples = [distribution_2d.sample(np.random.rand(2)) for _ in range(10000)]
    num_samples = np.zeros((values_num, values_num))
    for i, j in samples:
        num_samples[i][j] += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Distribution 2D")

    im1 = axes[0].imshow(
        values
        / values_num
        / values_num
        / distribution_2d.values_distribution.values_int,
        cmap="viridis",
        alpha=0.8,
    )
    axes[0].set_title("PDF")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(num_samples, cmap="viridis", alpha=0.8)
    axes[1].set_title("Sample Distribution")
    plt.colorbar(im2, ax=axes[1])


if __name__ == "__main__":
    sample_1d()
    sample_2d()
    plt.show()
