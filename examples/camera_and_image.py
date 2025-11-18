import numpy as np

from raytracing.shape import Cube, Sphere, Triangle
from raytracing.camera import Camera
from raytracing.ray_intersect_object import RayIntersectObject
from raytracing.transform import rotate_Y, rotate_Z, translate, scale
from raytracing.bounding_volume_hierarchy import BVH
from raytracing.bounding_box import AABB


def look_shape(camera: Camera, object: RayIntersectObject, n_samples=10):
    total_pixels = camera.film_height * camera.film_width
    n_processed = 0
    for i in range(camera.film_width):
        for j in range(camera.film_height):
            rgb = np.zeros(3)
            for _ in range(n_samples):
                view_ray = camera.generate_view_ray_from(i, j)
                if not object.ray_intersect(view_ray) is None:
                    rgb += np.ones(3)
            n_processed += 1
            print(
                "\rProgress %d%%" % (n_processed * 100 // total_pixels),
                end="",
                flush=True,
            )
            camera.write_to(i, j, rgb / n_samples)
    print("")


if __name__ == "__main__":
    camera = Camera(pos=np.array([0, 0, 5], dtype=np.float32))

    cube = Cube()
    cube.transform = rotate_Z(np.pi / 3)
    print("Generating an image for a cube...")
    look_shape(camera, cube, n_samples=3)
    camera.save_film("pics/", "cube")

    sphere = Sphere(10, 10)
    print("Generating an image for a sphere...")
    look_shape(camera, sphere, n_samples=3)
    camera.save_film("pics/", "sphere")

    triangles = [
        Triangle(
            np.array([0, 1, 0], dtype=np.float32),
            np.array([-1, -1, 0], dtype=np.float32),
            np.array([1, -1, 0], dtype=np.float32),
        )
        for _ in range(3)
    ]

    for i in range(3):
        triangles[i].transform = translate(-2 + 2 * i, 0, 0) @ rotate_Y(i * np.pi / 9)
    bvh = BVH(BVH.Type.MID_POINT, triangles)
    print("Generating an image for three triangles...")
    look_shape(camera, bvh, n_samples=3)
    camera.save_film("pics/", "triangles")

    spheres = []
    for _ in range(10):
        sphere = Sphere(10, 10)
        sphere.transform = translate(
            np.random.random() * 5, np.random.random() * 5, np.random.random() * 5
        ) @ scale(np.random.random(), np.random.random(), np.random.random())
        spheres.append(sphere)

    bvh = BVH(BVH.Type.MID_POINT, spheres)

    bbx = AABB()
    for sphere in spheres:
        bbx = AABB.union(bbx, sphere.bounding_box)
    camera.pos = np.array([0, 0, 2 * bbx.range_z().size()], dtype=np.float32)
    camera.look_at = bbx.center()
    print("Generating an image for ten spheres...")
    look_shape(camera, bvh, n_samples=3)
    camera.save_film("pics/", "spheres")
