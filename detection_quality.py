from __future__ import annotations
import cv2
import numpy as np


def _part_support(boxes, heads, bodies):
    if not heads or not bodies:
        return {}, {}
    head_boxes, body_boxes = boxes[heads], boxes[bodies]
    hs, bs = head_boxes[:, 2:] - head_boxes[:, :2], body_boxes[:, 2:] - body_boxes[:, :2]
    if hs[:, 1].min() > .4 * bs[:, 1].max() or hs[:, 1].max() < .025 * bs[:, 1].min():
        return {}, {}
    hc = (head_boxes[:, :2] + head_boxes[:, 2:]) * .5
    bc = (body_boxes[:, :2] + body_boxes[:, 2:]) * .5
    wr = hs[:, None, 0] / bs[None, :, 0]
    hr = hs[:, None, 1] / bs[None, :, 1]
    dx = abs(hc[:, None, 0] - bc[None, :, 0]) / bs[None, :, 0]
    dy = (head_boxes[:, None, 1] - body_boxes[None, :, 1]) / bs[None, :, 1]
    valid = (wr >= .1) & (wr <= .95) & (hr >= .025) & (hr <= .4) & (dx <= .55) & (dy >= -.2) & (dy <= .3)
    costs = np.where(valid, dx + .35 * abs(dy), np.inf)
    head_choices, body_choices = np.argmin(costs, axis=1), np.argmin(costs, axis=0)
    head_support = {h: bodies[int(choice)] for h, choice, valid in zip(heads, head_choices, np.isfinite(costs[np.arange(len(heads)), head_choices])) if valid}
    body_support = {b: heads[int(choice)] for b, choice, valid in zip(bodies, body_choices, np.isfinite(costs[body_choices, np.arange(len(bodies))])) if valid}
    return head_support, body_support


def select_detections(boxes, scores, names, *, iou=.65, head_iou=.3, limit=300):
    boxes, scores = np.asarray(boxes, dtype=np.float32), np.asarray(scores, dtype=np.float32)
    if not len(boxes):
        return []
    labels = np.asarray([str(name).casefold() for name in names])
    sizes = boxes[:, 2:] - boxes[:, :2]
    valid = np.isfinite(boxes).all(axis=1) & np.isfinite(scores) & (sizes > 0).all(axis=1)
    order = np.argsort(-scores, kind='stable')
    order = order[valid[order]][:3000]
    if len(order) > 512:
        accepted = set()
        for label in np.unique(labels[order]):
            indices = order[labels[order] == label]
            rects = boxes[indices].copy()
            rects[:, 2:] -= rects[:, :2]
            selected = cv2.dnn.NMSBoxes(rects.tolist(), scores[indices].tolist(), 0.0, .95)
            accepted.update(indices[np.asarray(selected, dtype=int).ravel()].tolist())
        order = np.asarray([index for index in order if index in accepted][:512], dtype=int)
    areas = sizes[:, 0] * sizes[:, 1]
    centers = (boxes[:, :2] + boxes[:, 2:]) * .5
    heads, bodies = ([int(i) for i in order if labels[i] == label] for label in ('head', 'body'))
    head_support, _ = _part_support(boxes, heads, bodies)

    def select(indices, threshold, support, head=False):
        if len(indices) < 2:
            return indices[:limit]
        current = np.asarray(indices, dtype=int)
        values = boxes[current]
        area = areas[current]
        width, height = sizes[current].T
        cx, cy = centers[current].T
        intersection = np.maximum(0, np.minimum(values[:, None, 2], values[None, :, 2]) - np.maximum(values[:, None, 0], values[None, :, 0])) * np.maximum(0, np.minimum(values[:, None, 3], values[None, :, 3]) - np.maximum(values[:, None, 1], values[None, :, 1]))
        containment = intersection / np.maximum(np.minimum(area[:, None], area[None, :]), 1e-12)
        dx, dy = abs(cx[:, None] - cx[None, :]), abs(cy[:, None] - cy[None, :])
        min_width, min_height = np.minimum(width[:, None], width[None, :]), np.minimum(height[:, None], height[None, :])
        near = (dx <= .25 * min_width) & (dy <= .25 * min_height)
        strict = (containment >= .98) & (dx <= .08 * min_width) & (dy <= .08 * min_height)
        overlaps = intersection / np.maximum(area[:, None] + area[None, :] - intersection, 1e-12)
        duplicates = (overlaps > threshold) | ((containment >= .9) & near)
        if support:
            parts = np.asarray([support.get(int(index), -1) for index in current])
            separate = (parts[:, None] >= 0) & (parts[None, :] >= 0) & (parts[:, None] != parts[None, :])
            if head:
                px, py = centers[parts].T
                pw, ph = sizes[parts].T
                distant_parts = (abs(px[:, None] - px[None, :]) >= .2 * np.minimum(pw[:, None], pw[None, :])) | (abs(py[:, None] - py[None, :]) >= .2 * np.minimum(ph[:, None], ph[None, :]))
                separate &= ((dx >= .4 * min_width) | (dy >= .4 * min_height)) & distant_parts
            else:
                shared = (parts[:, None] >= 0) & (parts[:, None] == parts[None, :])
                duplicates |= shared & (overlaps > .45)
            duplicates &= ~separate | strict
        suppressed = np.zeros(len(current), dtype=bool)
        kept = []
        for position, index in enumerate(current):
            if suppressed[position]:
                continue
            kept.append(int(index))
            suppressed |= duplicates[position]
            if len(kept) >= limit:
                break
        return kept

    kept_heads = select(heads, head_iou, head_support, head=True)
    _, body_support = _part_support(boxes, kept_heads, bodies)
    kept = kept_heads + select(bodies, iou, body_support)
    for label in np.unique(labels[order]):
        if label not in ('head', 'body'):
            kept.extend(select([int(i) for i in order if labels[i] == label], iou, {}))
    accepted = set(kept)
    return [int(i) for i in order if int(i) in accepted][:limit]
