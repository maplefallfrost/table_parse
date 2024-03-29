import heapq
from typing import List

import numpy as np

from modules.bounding_box import BoundingBox


def compute_horizontal_distance_threshold(bboxes: List[BoundingBox]) -> float:
    sorted_bboxes = sorted(bboxes, key=lambda x: (x.center[0], x.center[1]))
    horizontal_distances = [
        sorted_bboxes[i + 1].min_col - sorted_bboxes[i].max_col
        for i in range(len(sorted_bboxes) - 1)
    ]
    horizontal_distances = [dist for dist in horizontal_distances if dist > 0]
    threshold = np.percentile(horizontal_distances, 60)
    return threshold


def can_merge_by_projection(
    cluster: BoundingBox, bbox: BoundingBox, projection: np.ndarray
) -> bool:
    min_row = min(cluster.min_row, bbox.min_row)
    max_row = max(cluster.max_row, bbox.max_row)
    segment_projection = projection[min_row:max_row]
    return np.all(segment_projection > 0)


def can_merge_by_horizontal_distance(
    cluster: BoundingBox, bbox: BoundingBox, threshold: float
) -> bool:
    return (
        cluster.min_col - bbox.max_col < threshold
        and bbox.min_col - cluster.max_col < threshold
    )


def update_distances(
    heap: List[tuple[float, int, int]], bboxes: List[BoundingBox], new_bbox_index: int
):
    for i, bbox in enumerate(bboxes):
        if i != new_bbox_index and bbox is not None:
            dist = bboxes[new_bbox_index].min_distance(bbox)
            heapq.heappush(heap, (dist, new_bbox_index, i))


def clustering(bboxes: List[BoundingBox], image_height: int) -> List[BoundingBox]:
    sorted_bboxes = sorted(bboxes, key=lambda x: (x.center[0], x.center[1]))
    projection = np.zeros(image_height)
    for bbox in sorted_bboxes:
        projection[bbox.min_row : bbox.max_row] += 1
    horizontal_distance_threshold = compute_horizontal_distance_threshold(sorted_bboxes)

    heap = []
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            dist = bboxes[i].min_distance(bboxes[j])
            heapq.heappush(heap, (dist, i, j))

    while heap:
        dist, i, j = heapq.heappop(heap)
        if bboxes[i] is not None and bboxes[j] is not None:
            can_merge_flags = [
                can_merge_by_projection(bboxes[i], bboxes[j], projection),
                can_merge_by_horizontal_distance(
                    bboxes[i], bboxes[j], horizontal_distance_threshold
                ),
            ]
            if np.all(can_merge_flags):
                bboxes[i].merge(bboxes[j])
                bboxes.append(bboxes[i].copy())
                bboxes[i] = bboxes[j] = None
                update_distances(heap, bboxes, len(bboxes) - 1)

    return [bbox for bbox in bboxes if bbox is not None]
