from .constants import zero3f
from .ray_tracer import RayTracer


def render(
    ray_tracer: RayTracer,
    n_view_ray=1,
    name="direct light",
):
    camera = ray_tracer.scene.camera
    total_pixels = camera.film_height * camera.film_width
    processed_n = 0
    for row_i in range(camera.film_height):
        for col_i in range(camera.film_width):
            rgb = zero3f()
            for _ in range(n_view_ray):
                rgb += (
                    ray_tracer.render(
                        view_ray=camera.generate_view_ray_from(row_i, col_i)
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
