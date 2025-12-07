import numpy as np
from numpy.typing import NDArray


def uniform_sample_hemisphere(u: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Uniformly sample a point on the unit hemisphere.

    pdf(w) = 1 / (2pi)

    pdf(theta, phi) = p(w).(d(w) / (d(theta)d(phi))) = sin(theta) / (2pi)

    pdf(theta) = sin(theta), cdf(theta) = 1 - cos(theta)

    pdf(phi | theta) = 1 / (2pi), cdf(phi) = phi / (2pi)

    theta = arccos(1 - u[0]) or arccos(u[0]), phi = 2pi * u[1]
    """
    cos_theta = u[0]
    sin_theta = np.sqrt(np.max([0, 1 - cos_theta * cos_theta]))
    phi = 2 * np.pi * u[1]
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])


def uniform_hemisphere_pdf() -> np.float32:
    return 0.5 / np.pi


def uniform_sample_sphere(u: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Uniformly sample a point on the unit sphere.
    """
    cos_theta = 1 - 2 * u[0]
    sin_theta = np.sqrt(np.max([0, 1 - cos_theta * cos_theta]))
    phi = 2 * np.pi * u[1]
    return np.array(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], dtype=np.float32
    )


def uniform_sphere_pdf() -> np.float32:
    return 0.25 / np.pi


def uniform_sample_disk(u: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Uniformly sample a point on a unit disk.

    p(x, y) = 1 / pi

    p(r, theta) = p(x, y) J(x, y -> r, theta) = r / pi

    pdf(r) = 2r, cdf(r) = r^2

    pdf(theta | r) = 1 / (2pi), cdf(theta) = theta / (2pi)

    r = sqrt(u[0]), theta = 2pi * u[1]
    """
    r = np.sqrt(u[0])
    theta = 2 * np.pi * u[1]
    return np.array([r * np.cos(theta), r * np.sin(theta), 0], dtype=np.float32)


def uniform_disk_pdf():
    return 1 / np.pi


def concentric_sample_disk(u: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Concentricly sample a unit disk. Map points in the square x in [-1, 1] and
    y in [-1, 1] to the unit disk.

    r = x, theta = (y / x) * (pi / 4) when |y| < |x|

    r = y, theta = pi / 2 - (x / y) * (pi / 4) otherwise.
    """
    # Map u from [0, 1] to [-1, 1]
    u_offset = 2 * u - 1
    if np.isclose(u_offset[0], 0) and np.isclose(u_offset[1], 0):
        return np.zeros(2, dtype=np.float32)

    if np.abs(u_offset[1]) < np.abs(u_offset[0]):
        r = u_offset[0]
        theta = 0.25 * np.pi * u_offset[1] / u_offset[0]
    else:
        r = u_offset[1]
        theta = 0.5 * np.pi - 0.25 * np.pi * u_offset[0] / u_offset[1]
    return np.array([r * np.cos(theta), r * np.sin(theta), 0], dtype=np.float32)


def cosine_sample_hemisphere(u: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Cosine weighted sample a unit hemisphere such that the pdf of the sample is
    proportional to cos(theta).

    pdf(w) = c.cos(theta)

    cdf(hemisphere) = c.pi = 1 -> c = 1 / pi

    pdf(w) = cos(theta) / pi, p(theta, phi) = sin(theta) cos(theta) / pi

    If we first uniformly sample a unit disk with pdf(x, y) = 1 / pi, and
    map the point up to the hemisphere, we found that

    pdf(r, phi) = pdf(x, y) J(x, y -> r, phi) = pdf(x, y).r = r / pi

    pdf(theta, phi) = pdf(r, phi) J(r, phi -> theta, phi) = r / pi * cos(theta)
    = sin(theta)cos(theta) / pi

    The pdf is aligned with cosine weighted samples on the hemisphere.
    """
    disk_sample = uniform_sample_disk(u)
    x, y = disk_sample[0], disk_sample[1]
    z = np.sqrt(np.max([0, 1 - x * x - y * y]))
    return np.array([x, y, z], dtype=np.float32)


def cosine_sample_hemisphere_pdf(p: NDArray[np.float32]):
    normalized_p = normalize(p)
    return normalized_p[2] / np.pi


def uniform_sample_direction_in_cone(
    u: NDArray[np.float32], theta_max: np.float32
) -> NDArray[np.float32]:
    """
    Uniformly sample a direction within a cone whose half angle is theta_max.

    pdf(w) = c

    pdf(theta, phi) = sin(theta).pdf(w) = sin(theta).c

    cdf(theta_max, 2.pi) = 2.pi.(1 - cos(theta_max)).c = 1 ->
    c = 1 / (2.pi.(1 - cos(theta_max)))

    pdf(theta) = sin(theta) / (1 - cos(theta_max))

    pdf(phi | theta) = 1 / (2.pi)

    cdf(theta) = (1 - cos(theta)) / (1 - cos(theta_max)),
    cos(theta) = 1 - (1 - cos(theta_max)).u[0]

    phi = 2.pi.u[1]
    """
    cos_theta = 1 - (1 - np.cos(theta_max)) * u[0]
    sin_theta = np.sqrt(np.max([0, 1 - cos_theta * cos_theta]))
    phi = 2 * np.pi * u[1]
    return np.array(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], dtype=np.float32
    )


def uniform_sample_cone_pdf(theta_max: np.float32):
    return 0.5 / (np.pi * (1 - np.cos(theta_max)))


def uniform_sample_triangle(u: NDArray[np.float32], vertices) -> NDArray[np.float32]:
    """
    Uniformly sammple a point on a triangle with given vertices.

    Suppose the sampled points p expressed by the barycentric coordinates is
    (alpha, beta), such that p(alpha, beta) = alpha * v[0] + beta * v[1] +
    (1 - alpha -beta) * v[2].

    As we know, pdf(p) = 1 / area.

    dp / d(alpha) = v[0] - v[2], dp / d(beta) = v[1] - v[2]

    dA = (v[0] - v[2]) x (v[1] - v[2]) d(alpha).d(beta) = 2.area.d(alpha)d(beta)

    pdf(alpha, beta) = 2.area.pdf(p) = 2.

    pdf(alpha) = 2(1 - alpha), pdf(beta | alpha) = 1 / (1 - alpha).

    cdf(alpha) = 2alpha - alpha^2. cdf(beta | alpha) =  beta / (1 - alpha).

    2alpha - alpha^2 = u[0], beta / (1 - alpha) = u[1]

    alpha^2 - 2alpha + 1 = 1 - u[0], (alpha - 1)^2 = 1 - u'[0],
    alpha = 1 - sqrt(1 - u'[0]) <-> alpha = 1 - sqrt(u[0]).

    beta = sqrt(u[0])u[1].
    """
    alpha = 1 - np.sqrt(u[0])
    beta = (1 - alpha) * u[1]
    return alpha * vertices[0] + beta * vertices[1] + (1 - alpha - beta) * vertices[2]


def uniform_sample_triangle_pdf(vertices):
    e1 = vertices[1] - vertices[0]
    e2 = vertices[2] - vertices[0]
    return 2 / np.cross(e1, e2)
