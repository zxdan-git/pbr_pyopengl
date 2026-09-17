import numpy as np

from raytracing.bounding_volume_hierarchy import BVH
from raytracing.camera import Camera
from raytracing.ray_tracers.path_tracer import PathTracer
from raytracing.constants import red, green, blue, white
from raytracing.lights.point_light import PointLight
from raytracing.bxdfs.lambertian import Lambertain
from raytracing.materials.material_one import MaterialOne
from raytracing.render_util import render
from raytracing.scene import Scene
from raytracing.shapes.sphere import Sphere
from raytracing.shapes.cube import Cube
from raytracing.shapes.triangle import Triangle
from raytracing.shapes.triangle_mesh import TriangleMesh
from raytracing.transform import translate, rotate_Y
from raytracing.typing import vec3f, array_f, array_u

if __name__ == "__main__":
    camera = Camera(pos=np.array([0, 0, 3]), look_at=np.array([0, 0, 0]))
    point_light = PointLight(intensity=np.ones(3) * 2, pos=vec3f(0, 0, 0))
    point_light.n_samples = 1

    square_1 = TriangleMesh(
        vertex=array_f([[0, 1, 1], [0, -1, 1], [0, -1, -1], [0, 1, -1]]),
        tex_coord=array_f([[0, 0], [0, 1], [1, 1], [1, 0]]),
        face_index=array_u([0, 1, 2, 0, 2, 3]),
        tex_index=array_u([0, 1, 2, 0, 2, 3]),
    )
    square_1.transform = translate(-1, 0, 0)

    square_2 = square_1.copy()
    square_2.transform = translate(1, 0, 0) @ rotate_Y(np.pi)
    square_2.material = MaterialOne(Lambertain(), red())

    scene = Scene(
        camera,
        [square_1, square_2],
        [point_light],
    )
    scene.setup_bvh(BVH.Type.SAH)
    path_tracer = PathTracer(scene, max_path_len=5)
    render(
        path_tracer,
        1,
        "indirect light 1 sample per pixel",
    )
