import numpy as np
from typing import List

from ..bounding_box import AABB
from ..interval import Interval
from ..ray_intersect_object import RayIntersectObject
from .surface_area_heuristic_util import BucketInfo, min_sah_partition
from ..util import partition, partition_cmp, nth_element


def mid_point_partition_on_target_axis(
    objects: List[RayIntersectObject], start, end, target_axis, target_range: Interval
):
    mid_point = target_range.to_array().mean()
    return partition_cmp(
        objects,
        mid_point,
        start,
        end,
        lambda object: object.bounding_box.center()[target_axis],
    )


def equal_partition_on_target_axis(
    objects: List[RayIntersectObject], start, end, target_axis, target_range: Interval
):
    return nth_element(
        objects,
        (end - start) // 2,
        start,
        end,
        lambda object: object.bounding_box.center()[target_axis],
    )


def sah_partition_on_target_aixs(
    objects: List[RayIntersectObject], start, end, target_axis, target_range: Interval
):
    # Build up buckets according to the center position on target axis.
    bucket_range = 12
    buckets: List[BucketInfo] = [BucketInfo() for _ in range(bucket_range + 1)]
    for object in objects[start:end]:
        idx = int(
            np.clip(
                bucket_range
                * (object.bounding_box.center()[target_axis] - target_range.lower)
                / target_range.size(),
                0,
                bucket_range,
            )
        )
        buckets[idx].objects.append(object)
        buckets[idx].total_cost += object.ray_intersect_cost()
        buckets[idx].bbx = AABB.union(buckets[idx].bbx, object.bounding_box)

    left, _ = min_sah_partition(buckets)
    return partition(
        objects,
        start,
        end,
        lambda object: object in left.objects,
    )
