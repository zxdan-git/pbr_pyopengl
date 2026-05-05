import numpy as np
from typing import List

from raytracing.bounding_volume_hierarchy import BVH
from raytracing.camera import Camera
from raytracing.direct_ray_tracer import DirectRayTracer
from raytracing.lights.area_light import AreaLight
from raytracing.lights.point_light import PointLight
from raytracing.lights.spot_light import SpotLight
from raytracing.light import Light
from raytracing.bxdfs.lambertian import Lambertain
from raytracing.materials.material_one import MaterialOne
from raytracing.ray import Ray
from raytracing.ray_intersect_object import RayIntersectObject
from raytracing.shapes.sphere import Sphere
from raytracing.shapes.cube import Cube
from raytracing.shapes.triangle import Triangle
from raytracing.transform import translate, scale


def render(
    camera: Camera,
    objects: List[RayIntersectObject],
    lights: List[Light],
    sample_strategy: DirectRayTracer.SampleStrategy,
    n_view_ray=1,
    name="direct light",
):
    direct_ray_tracer = DirectRayTracer(objects, lights)
    total_pixels = camera.film_height * camera.film_width
    processed_n = 0
    for row_i in range(camera.film_height):
        for col_i in range(camera.film_width):
            rgb = np.zeros(3)
            for _ in range(n_view_ray):
                rgb += (
                    direct_ray_tracer.render(
                        view_ray=camera.generate_view_ray_from(row_i, col_i),
                        sample_strategy=sample_strategy,
                    )
                    / n_view_ray
                )
            camera.write_to(row_i, col_i, rgb)
            processed_n += 1
            print(
                "\rProgress %d%%" % (processed_n * 100 // total_pixels),
                end="",
                flush=True,
            )
    print("")
    camera.save_film("pics", name)
    camera.show_film()


if __name__ == "__main__":
    camera = Camera(pos=np.array([0, 5, 5]), look_at=np.array([0, 3, 0]))
    point_light = PointLight(intensity=np.ones(3) * 200, pos=np.array([0, 10, 0]))
    spot_light_red = SpotLight(
        intensity=np.array([1, 0, 0], dtype=np.float32) * 100,
        pos=np.array([8, 8, 8]),
        dir=np.array([-1, -1, -1]),
        theta_max=np.pi / 3,
        theta_decay=np.pi / 4,
    )

    spot_light_green = SpotLight(
        intensity=np.array([0, 1, 0], dtype=np.float32) * 100,
        pos=np.array([-8, 8, 8]),
        dir=np.array([1, -1, -1]),
        theta_max=np.pi / 3,
        theta_decay=np.pi / 4,
    )

    spot_light_blue = SpotLight(
        intensity=np.array([0, 0, 1], dtype=np.float32) * 100,
        pos=np.array([0, 8 * np.sqrt(2), 0]),
        dir=np.array([0, -1, 0]),
        theta_max=np.pi / 3,
        theta_decay=np.pi / 4,
    )
    sphere = Sphere(10, 10)
    sphere.material = MaterialOne(Lambertain())
    sphere.transform = translate(0, 2, 0)
    cube = Cube()
    cube.transform = scale(5, 0.1, 5)
    cube.material = MaterialOne(Lambertain())
    triangle = Triangle(np.array([0, 0.2, 0]), np.array([0, 0, 1]), np.array([1, 0, 0]))
    triangle.transform = translate(-0.2, 3.5, 0)
    triangle.material = MaterialOne(Lambertain())
    sphere_light = AreaLight(intensity=np.ones(3) * 200, shape=Sphere(10, 10))
    sphere_light.n_samples = 1
    sphere_light.transform = translate(0, 10, 0)
    triangle_light = AreaLight(
        intensity=np.ones(3) * 10,
        shape=Triangle(
            np.array([-5, 10, 0]), np.array([5, 10, 0]), np.array([0, 10, 5])
        ),
    )
    triangle_light.n_samples = 3
    bvh = BVH(BVH.Type.MID_POINT, [sphere, cube, triangle])
    render(
        camera,
        [bvh],
        [spot_light_red, spot_light_green, spot_light_blue],
        DirectRayTracer.SampleStrategy.UNIFORM_SAMPLE_ALL,
        1,
        "sphere light sample all 1 view ray bvh",
    )
