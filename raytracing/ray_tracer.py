from abc import ABC, abstractmethod

from .scene import Scene
from .ray import Ray
from .typing import Vec3f


class RayTracer(ABC):
    def __init__(self, scene: Scene):
        self.scene = scene

    @abstractmethod
    def render(self, view_ray: Ray) -> Vec3f:
        pass
