import numpy as np

from raytracing.shape import Triangle, Cube, Sphere
from raytracing.transform import translate, rotate_X, rotate_Y, rotate, scale


def test_shape_update_area():
    # Test 1: non-transformed triangle.
    print("Test 1: non-transformed triangle.")
    triangle = Triangle(np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([-7, 8, -9]))
    assert np.isclose(34.72751070837067, triangle.area)

    # Test 2: non-transformed cube.
    print("Test 2: non-transformed cube.")
    cube = Cube()
    assert np.isclose(24, cube.area)

    # Test 3: non-transformed sphere.
    print("Test 3: non-transformed sphere.")
    sphere = Sphere(300, 300)
    assert np.isclose(4 * 3.1415, sphere.area, rtol=5e-3, atol=1e-3)

    # Test 4: translated triangle.
    print("Test 4: translated triangle.")
    triangle.transform = translate(1, 2, 3)
    assert np.isclose(34.72751070837067, triangle.area)

    # Test 5: translated cube.
    print("Test 5: translated cube.")
    cube.transform = translate(4, 5, 6)
    assert np.isclose(24, cube.area)

    # Test 6: translated sphere.
    print("Test 6: translated sphere.")
    sphere.transform = translate(7, 8, 9)
    assert np.isclose(4 * 3.1415, sphere.area, rtol=5e-3, atol=1e-3)

    # Test 7: rotated triangle.
    print("Test 7: tra triangle.")
    triangle.transform = rotate_X(np.pi / 3)
    assert np.isclose(34.72751070837067, triangle.area)

    # Test 8: rotated cube.
    print("Test 8: rotated cube.")
    cube.transform = rotate_Y(np.pi / 4)
    assert np.isclose(24, cube.area)

    # Test 9: rotated sphere.
    print("Test 9: rotated sphere.")
    sphere.transform = rotate(np.array([1, 1, 1]), np.pi / 6)
    assert np.isclose(4 * 3.1415, sphere.area, rtol=5e-3, atol=1e-3)

    # Test 10: scaled triangle.
    print("Test 10: scaled triangle.")
    triangle.transform = scale(2, 2, 2)
    assert np.isclose(34.72751070837067 * 4, triangle.area)

    # Test 11: scaled cube.
    print("Test 11: scaled cube.")
    cube.transform = scale(2, 3, 4)
    assert np.isclose(208, cube.area)

    # Test 12: scaled sphere.
    print("Test 12: scaled sphere.")
    sphere.transform = scale(2, 2, 2)
    assert np.isclose(16 * 3.1415, sphere.area, rtol=5e-3, atol=1e-3)


if __name__ == "__main__":
    test_shape_update_area()
