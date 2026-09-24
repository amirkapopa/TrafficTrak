"""Deterministic test double: turns white rectangles drawn on a black frame
into 'car' detections.  Used to test the causal risk estimator without model
weights; the production detector is YOLOX."""

import cv2
import numpy as np

from src.detection import COCO_NAMES, Detections


class RectangleDetector:
    on_gpu = False

    def __call__(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h >= 50:
                boxes.append([x, y, x + w, y + h])
        if not boxes:
            return Detections.empty()
        boxes = np.array(sorted(boxes), dtype=np.float32)
        return Detections(boxes, np.full(len(boxes), 0.9, np.float32), np.full(len(boxes), COCO_NAMES.index("car"), np.int64))


def render(rects, size=(640, 360)):
    img = np.zeros((size[1], size[0], 3), np.uint8)
    for x0, y0, x1, y1 in rects:
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), -1)
    return img
