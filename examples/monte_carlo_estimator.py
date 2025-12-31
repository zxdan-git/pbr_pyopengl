import numpy as np
import matplotlib.pyplot as plt


class Sampler:
    def __init__(self):
        self.start = 0
        self.end = 0

    def set_domain(self, start, end):
        self.start = start
        self.end = end

    def sample(self, u):
        pass

    def pdf(self, v):
        pass


class UniformSampler(Sampler):
    def sample(self, u):
        return self.start + u * (self.end - self.start)

    def pdf(self, v):
        if self.start <= v <= self.end:
            return 1 / (self.end - self.start)
        return 0


class CosineSampler(Sampler):
    def set_domain(self, start, end):
        super().set_domain(start, end)
        self.sin_start = np.sin(start)
        self.sin_end = np.sin(end)

    def sample(self, u):
        return np.asin((1 - u) * self.sin_start + u * self.sin_end)

    def pdf(self, v):
        if self.start <= v <= self.end:
            return np.cos(v) / (self.sin_end - self.sin_start)
        return 0


def monte_carlo(func, sampler):
    est = []
    for _ in range(1000):
        x = sampler.sample(np.random.rand())
        est.append(func(x) / sampler.pdf(x))
    return est


def estimate(func, int_func, sampler: Sampler, start, end, title):
    interval = (end - start) / 3
    x = np.arange(start, end, interval) + interval

    # Collect true values, means, and variances for each xi
    true_values = []
    means = []
    variances = []

    for xi in x:
        sampler.set_domain(start, xi)
        est = monte_carlo(func, sampler)
        true_values.append(int_func(xi) - int_func(start))
        means.append(np.mean(est))
        variances.append(np.var(est))

    # Create a single plot
    _, ax = plt.subplots(figsize=(12, 8))

    # Plot the true integral values
    ax.plot(
        x,
        true_values,
        "b-",
        linewidth=2,
        label="True integral",
        marker="o",
        markersize=10,
    )

    # Plot the mean estimates
    ax.plot(
        x,
        means,
        "g--",
        linewidth=1.5,
        label="Mean of estimates",
        marker="s",
        markersize=8,
    )

    # Add error bars showing variance (standard deviation)
    std_devs = np.sqrt(variances)
    ax.errorbar(
        x,
        means,
        yerr=std_devs,
        fmt="none",
        ecolor="red",
        capsize=5,
        alpha=0.6,
        label="±1 std dev (√variance)",
    )

    # Add shaded region for variance
    ax.fill_between(
        x,
        np.array(means) - std_devs,
        np.array(means) + std_devs,
        alpha=0.2,
        color="red",
        label="Variance region",
    )

    ax.set_xlabel("xi", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()


if __name__ == "__main__":
    estimate(
        func=lambda x: 2 * x,
        int_func=lambda x: x * x,
        sampler=UniformSampler(),
        start=2,
        end=8,
        title="estimate integral of 2x with uniform sampling",
    )
    estimate(
        func=lambda x: np.cos(x),
        int_func=lambda x: np.sin(x),
        sampler=UniformSampler(),
        start=2,
        end=8,
        title="estimate integral of cos(x) with uniform sampling",
    )
    estimate(
        func=lambda x: 2 * x,
        int_func=lambda x: x * x,
        sampler=CosineSampler(),
        start=0,
        end=0.5 * np.pi,
        title="estimate integral of 2x with cosine sampling",
    )
    estimate(
        func=lambda x: np.cos(x),
        int_func=lambda x: np.sin(x),
        sampler=CosineSampler(),
        start=0,
        end=0.5 * np.pi,
        title="estimate integral of cos(x) with cosine sampling",
    )
    plt.show()
