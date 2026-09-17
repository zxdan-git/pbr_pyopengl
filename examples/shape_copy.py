import numpy as np

from raytracing.bounding_volume_hierarchy import BVH
from raytracing.camera import Camera
from raytracing.ray_tracers.direct_ray_tracer import DirectRayTracer
from raytracing.lights.spot_light import SpotLight
from raytracing.bxdfs.lambertian import Lambertain
from raytracing.materials.material_one import MaterialOne
from raytracing.render_util import render
from raytracing.scene import Scene
from raytracing.shapes.sphere import Sphere
from raytracing.shapes.cube import Cube
from raytracing.shapes.triangle import Triangle
from raytracing.shapes.triangle_mesh import TriangleMesh
from raytracing.transform import translate, scale

if __name__ == "__main__":
    camera = Camera(
        pos=np.array([0, 8, 2]), look_at=np.array([0, 0, 2]), up=np.array([0, 0, -1])
    )
    spot_light_red = SpotLight(
        intensity=np.array([1, 0, 0], dtype=np.float32) * 100,
        pos=np.array([8, 8, 0]),
        dir=np.array([-1, -1, 0]),
        theta_max=np.pi / 3,
        theta_decay=np.pi / 4,
    )

    spot_light_green = SpotLight(
        intensity=np.array([0, 1, 0], dtype=np.float32) * 100,
        pos=np.array([-8, 8, 0]),
        dir=np.array([1, -1, 0]),
        theta_max=np.pi / 3,
        theta_decay=np.pi / 4,
    )

    print("creating sphere...")
    sphere = Sphere(10, 10)
    sphere.material = MaterialOne(Lambertain())
    sphere.transform = translate(0, 0, -2)
    print("copying sphere...")
    sphere_copy = sphere.copy()
    sphere = None

    print("creating cube...")
    cube = Cube()
    cube.transform = scale(0.5, 0.5, 0.5)
    cube.material = MaterialOne(Lambertain())
    print("copying cube...")
    cube_copy = cube.copy()
    cube = None

    print("creating triangle...")
    triangle = Triangle(np.array([0, 0.2, 0]), np.array([0, 0, 1]), np.array([1, 0, 0]))
    triangle.transform = translate(0, 0, 2)
    triangle.material = MaterialOne(Lambertain())
    print("copying triangle...")
    triangle_copy = triangle.copy()
    triangle = None

    print("creating mesh...")
    mesh = TriangleMesh.create_mesh_from_obj("models/bunny.obj")
    mesh.transform = translate(0, 1, 4) @ scale(15, 15, 15)
    print("copying mesh...")
    mesh_copy = mesh.copy()
    mesh = None

    scene = Scene(
        camera,
        [
            sphere_copy,
            cube_copy,
            triangle_copy,
            mesh_copy,
        ],
        [spot_light_red, spot_light_green],
    )
    scene.setup_bvh(BVH.Type.SAH)
    direct_ray_tracer = DirectRayTracer(
        scene, DirectRayTracer.SampleStrategy.UNIFORM_SAMPLE_ALL
    )
    render(
        direct_ray_tracer,
        1,
        "shape copy",
    )
