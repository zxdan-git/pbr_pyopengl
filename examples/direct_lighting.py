import numpy as np

from raytracing.camera import Camera
from raytracing.direct_ray_tracer import DirectRayTracer
from raytracing.lights.point_light import PointLight
from raytracing.bxdfs.lambertian import Lambertain
from raytracing.materials.material_one import MaterialOne
from raytracing.shape import Cube, Sphere
from raytracing.transform import translate, scale

if __name__ == "__main__":
    camera = Camera(pos=np.array([0, 5, 5]))
    point_light = PointLight(le=np.ones(3) * 200, pos=np.array([0, 10, 0]))
    sphere = Sphere(10, 10)
    sphere.material = MaterialOne(Lambertain())
    sphere.transform = translate(0, 2, 0)
    cube = Cube()
    cube.transform = scale(5, 0.1, 5)
    cube.material = MaterialOne(Lambertain())
    direct_ray_tracer = DirectRayTracer(objects=[cube, sphere], lights=[point_light])
    view_ray_n = 3
    total_pixels = camera.film_height * camera.film_width
    processed_n = 0
    for row_i in range(camera.film_height):
        for col_i in range(camera.film_width):
            rgb = np.zeros(3)
            for _ in range(view_ray_n):
                rgb += (
                    direct_ray_tracer.render(
                        view_ray=camera.generate_view_ray_from(row_i, col_i),
                        sample_strategy=DirectRayTracer.SampleStrategy.UNIFORM_SAMPLE_ALL,
                    )
                    / view_ray_n
                )
            camera.write_to(row_i, col_i, rgb)
            processed_n += 1
            print(
                "\rProgress %d%%" % (processed_n * 100 // total_pixels),
                end="",
                flush=True,
            )
    print("")
    camera.save_film("pics", "direct_light")
    camera.show_film()
