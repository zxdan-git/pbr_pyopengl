from typing import List

from .bounding_volume_hierarchy import BVH
from .camera import Camera
from .constants import zero3f
from .intersection import Intersection
from .light import Light
from .ray import Ray
from .ray_intersect_object import RayIntersectObject


class Scene:
    def __init__(
        self,
        camera: Camera = None,
        objects: List[RayIntersectObject] = [],
        lights: List[Light] = [],
    ):
        self.camera: Camera = camera
        self.objects: List[RayIntersectObject] = objects
        self.lights: List[Light] = lights
        self.bvh: BVH = None

    def setup_bvh(self, type: BVH.Type):
        if len(self.objects) == 0:
            return
        self.bvh = BVH(type, self.objects)

    def ray_intersect_objects(self, ray: Ray) -> Intersection:
        if not self.bvh is None:
            return self.bvh.ray_intersect(ray)
        inter = None
        for object in self.objects:
            inter_i = object.ray_intersect(ray)
            if not inter_i is None:
                inter = inter_i
        return inter

    def light_le(self, view_ray: Ray):
        le = zero3f()
        for light in self.lights:
            if light.is_delta():
                continue
            le += light.le(view_ray)
        return le
