import numpy as np

from raytracing.bounding_volume_hierarchy import BVH
from raytracing.camera import Camera
from raytracing.lights.point_light import PointLight
from raytracing.lights.spot_light import SpotLight
from raytracing.shapes.cube import Cube
from raytracing.shapes.sphere import Sphere
from raytracing.shapes.triangle_mesh import TriangleMesh
from raytracing.ray_tracers.direct_ray_tracer import DirectRayTracer
from raytracing.render_util import render
from raytracing.scene import Scene
from raytracing.transform import translate, scale

if __name__ == "__main__":
    camera = Camera(pos=np.array([5, 5, 5]), look_at=np.array([0, 0, 0]))
    point_light = PointLight(intensity=np.ones(3) * 200, pos=np.array([8, 8, 8]))
    spot_light = SpotLight(
        intensity=np.ones(3) * 100,
        pos=np.array([0, 10, 0]),
        dir=np.array([0, -1, 0]),
        theta_max=np.pi / 4,
        theta_decay=np.pi / 6,
    )

    sphere = Sphere(10, 10)
    sphere.transform = translate(0, 2, 0)
    cube = Cube()
    cube.transform = scale(5, 0.5, 5)

    sphere_mesh = TriangleMesh(
        sphere.vertex, sphere.tex_coord, sphere.face_index, sphere.tex_index
    )
    sphere_mesh.transform = translate(0, 2, 0)

    cube_mesh = TriangleMesh(
        cube.vertex, cube.tex_coord, cube.face_index, cube.tex_index
    )
    cube_mesh.transform = scale(3, 0.5, 3)

    obj_mesh = TriangleMesh.create_mesh_from_obj("models/bunny.obj")
    obj_mesh.transform = translate(0, 1, 0) @ scale(15, 15, 15)

    scene = Scene(
        camera,
        [obj_mesh, cube],
        [spot_light],
    )
    scene.setup_bvh(BVH.Type.SAH)
    direct_ray_tracer = DirectRayTracer(
        scene, DirectRayTracer.SampleStrategy.UNIFORM_SAMPLE_ALL
    )
    render(
        direct_ray_tracer,
        1,
        "obj mesh spot light sample all 1 view ray bvh",
    )
