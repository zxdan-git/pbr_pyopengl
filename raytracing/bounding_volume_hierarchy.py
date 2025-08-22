import numpy as np
from raytracing.bounding_box import AABB
from raytracing.shape import Shape
from raytracing.util import partition
from raytracing.ray import Ray, RayIntersectObject
from typing import List


class BVH(RayIntersectObject):
    def __init__(self, shapes):
        self.__shapes: List[Shape] = shapes
        self.__root: BVH.Node = self.build(0, len(shapes))

    def build(self, start, end):
        node = BVH.Node()
        if end - start == 1:
            node.shapes.append(self.__shapes[start])
            node.bbx = self.__shapes[start].bounding_box
            return node

        centroid_bbx = AABB()
        for i in range(start, end):
            centroid_bbx.embrace(self.__shapes[i].centroid())

        target_axis = centroid_bbx.get_max_axis()
        target_range = centroid_bbx.get_range(target_axis)

        """
        If all objects have the same centroid, stop partition and save them
        in one node. Otherwise the parition will not end.
        """
        if np.isclose(target_range.lower, target_range.upper):
            for shape in self.__shapes[start:end]:
                node.shapes.append(shape)
        else:
            pivot = target_range.to_array().mean()
            mid = partition(
                self.__shapes,
                pivot,
                start,
                end,
                lambda shape: shape.centroid()[target_axis],
            )
            node.left = self.build(start, mid)
            node.right = self.build(mid, end)

        for shape in self.__shapes[start:end]:
            node.bbx = AABB.union(node.bbx, shape.bounding_box)
        return node

    def ray_intersect(self, ray: Ray):
        return self.__root.ray_intersect(ray)

    @property
    def root(self):
        return self.__root

    class Node(RayIntersectObject):
        def __init__(self):
            self.left: BVH.Node = None
            self.right: BVH.Node = None
            self.bbx = AABB()
            self.shapes: List[Shape] = []

        def ray_intersect(self, ray: Ray):
            if not self.bbx.ray_intersect(ray):
                return None

            if len(self.shapes):
                intersect = False
                for shape in self.shapes:
                    if not shape.ray_intersect(ray) is None:
                        intersect = True
                """
                Note: we should return ray.t_max to get the nearest
                intersection.
                """
                if intersect:
                    return ray.t_max
                else:
                    return None

            for child in [self.left, self.right]:
                if not child is None:
                    intersect = child.ray_intersect(ray)
                    if not intersect is None:
                        return intersect
            return None
