from collections import deque
import math
import time

import cv2
import numpy as np


class OverlayPixelCleaner:
    def __init__(self, full_width, full_height, region):
        self.width, self.height = full_width, full_height
        self.region = region
        self._history = deque(maxlen=3)
        self._regions = ()
        self._geometry = None

    def set_regions(self, regions):
        valid = []
        for item in regions[:256]:
            if len(item) != 6:
                continue
            x1, y1, x2, y2, color, outline = item
            if color not in (0, 1) or not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                continue
            x1, y1, x2, y2 = max(0., x1), max(0., y1), min(1., x2), min(1., y2)
            if x2 > x1 and y2 > y1:
                valid.append((x1, y1, x2, y2, int(color), bool(outline)))
        current = tuple(valid)
        if current != self._regions:
            if self._history and self._regions:
                self._history[-1] = (time.perf_counter()+.1, self._regions)
            if current:
                self._history.append((math.inf, current))
            self._geometry = None
        self._regions = current

    def _masks(self, shape):
        now = time.perf_counter()
        while self._history and now > self._history[0][0]:
            self._history.popleft()
            self._geometry = None
        if self._geometry is None or self._geometry[0].shape != shape:
            masks = [np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)]
            rx, ry, _, _ = self.region
            for _, regions in self._history:
                for x1, y1, x2, y2, color, outline in regions:
                    left, top = round(x1*self.width-rx), round(y1*self.height-ry)
                    right, bottom = round(x2*self.width-rx), round(y2*self.height-ry)
                    if not outline:
                        left,top,right,bottom=left-2,top-2,right+2,bottom+2
                    cv2.rectangle(masks[color], (left, top), (right, bottom), 255, 7 if outline else -1)
            self._geometry = masks
        return self._geometry

    def clean(self, frame):
        if not self._history:
            return frame
        masks = self._masks(frame.shape[:2])
        colors = ((188, 239, 178), (252, 185, 128))
        selected = np.zeros(frame.shape[:2], np.uint8)
        for color, area in zip(colors, masks):
            low, high = (color + (0,), color + (255,)) if frame.shape[2] == 4 else (color, color)
            pixels = cv2.inRange(frame, low, high)
            cv2.bitwise_and(pixels, area, dst=pixels)
            cv2.bitwise_or(selected, pixels, dst=selected)
        if not cv2.countNonZero(selected):
            return frame
        x, y, w, h = cv2.boundingRect(selected)
        left, top = max(0, x-8), max(0, y-8)
        right, bottom = min(frame.shape[1], x+w+8), min(frame.shape[0], y+h+8)
        bgr = np.ascontiguousarray(frame[top:bottom, left:right, :3])
        cleaned = cv2.inpaint(bgr, selected[top:bottom, left:right], 2, cv2.INPAINT_NS)
        result = frame.copy()
        result[top:bottom, left:right, :3] = cleaned
        return result
