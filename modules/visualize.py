from typing import List

import cv2
import numpy as np


def visualize_bounding_boxes(
    image: np.ndarray, bboxes: List[List[int]], output_path: str
):
    for bbox in bboxes:
        pts = np.array(bbox, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.imwrite(output_path, image)
