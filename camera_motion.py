from collections import deque
from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraEstimate:
    shift: tuple
    quality: float


class CameraMotion:
    def __init__(self):
        self._window = None

    def estimate(self, previous, current, boxes=(), *, hint=None):
        if previous.shape != current.shape or previous.ndim != 2 or min(previous.shape) < 32:
            return None
        height, width = current.shape
        size = (min(128, width), min(96, height))
        if self._window is None or self._window.shape != size[::-1]:
            self._window = cv2.createHanningWindow(size, cv2.CV_32F)
        a = cv2.resize(previous, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        b = cv2.resize(current, size, interpolation=cv2.INTER_AREA).astype(np.float32)
        shift, response = cv2.phaseCorrelate(a, b, self._window)
        dx, dy = shift[0]*width/size[0], shift[1]*height/size[1]
        candidates = []
        if (all(math.isfinite(v) for v in (dx, dy, response)) and response >= .25
                and abs(dx) <= width*.45 and abs(dy) <= height*.45):
            candidates.append((dx, dy, response))
        if (hint is not None and all(math.isfinite(v) for v in hint)
                and abs(hint[0]) <= width*.45 and abs(hint[1]) <= height*.45):
            candidates.append((*hint, .5))
        candidates.append((0., 0., .5))
        for dx, dy, response in candidates:
            found = self._validate(previous, current, boxes, dx, dy, response)
            if found is not None:
                return found
        return None

    @staticmethod
    def _validate(previous, current, boxes, dx, dy, response):
        height, width = current.shape
        dx, dy = (0. if abs(v) < .5 else v for v in (dx, dy))
        ix, iy = round(dx), round(dy)
        side = min(18, width//8, height//8)
        positions, sources, targets = [], [], []
        for row, fy in enumerate((.18, .5, .82)):
            for col, fx in enumerate((.12, .38, .62, .88)):
                x, y = round(width*fx-side/2), round(height*fy-side/2)
                if (x+ix < 0 or y+iy < 0 or x+ix+side > width or y+iy+side > height
                        or any(x < bx2 and x+side > bx1 and y < by2 and y+side > by1
                               for bx1, by1, bx2, by2 in boxes)):
                    continue
                if abs(x+side/2-width/2) < side and abs(y+side/2-height/2) < side:
                    continue
                sources.append(previous[y:y+side, x:x+side].reshape(-1))
                targets.append(current[y+iy:y+iy+side, x+ix:x+ix+side].reshape(-1))
                positions.append((row, col))
        if len(sources) < 3:
            return None
        a, b = np.asarray(sources, dtype=np.float32), np.asarray(targets, dtype=np.float32)
        errors = np.mean(np.abs(a-b), axis=1)
        ac, bc = a-a.mean(axis=1, keepdims=True), b-b.mean(axis=1, keepdims=True)
        av, bv = np.einsum('ij,ij->i', ac, ac), np.einsum('ij,ij->i', bc, bc)
        denominator = np.sqrt(av*bv)
        correlation = np.divide(np.einsum('ij,ij->i', ac, bc), denominator,
                                out=np.zeros_like(denominator), where=denominator > 1e-8)
        textured = av >= 49*side*side
        eligible = int(np.count_nonzero(textured))
        accepted = textured & (correlation >= .75) & (errors <= 26)
        matched = [p for p, selected in zip(positions, accepted) if selected]
        if (len(matched) < 3 or len(matched) < eligible*.55
                or len({p[0] for p in matched}) < 2 or len({p[1] for p in matched}) < 2):
            return None
        return CameraEstimate((float(dx), float(dy)), min(1., response, len(matched)/max(1, eligible)))


class MouseCalibration:
    def __init__(self):
        self.generation = None
        self.reset()

    def reset(self):
        self.scales = (None, None)
        self.delays = (None, None)
        self.delay = 0.
        self._samples = deque(maxlen=64)
        self._fitted_at = -math.inf
        self._input_status = {}
        self._input_status_at = -math.inf

    @property
    def state(self):
        return dict(generation=self.generation, scales=self.scales, delay=self.delay, delays=self.delays,
                    input=self._input_status, fit_samples=len(self._samples))

    def project(self, snapshot, start, end):
        if (snapshot is None or not snapshot.active or snapshot.generation != self.generation
                or not any(v is not None for v in self.scales)):
            return 0., 0.
        result, intervals = [], {}
        for axis, scale in enumerate(self.scales):
            if scale is None:
                result.append(0.)
                continue
            delay = self.delays[axis] if self.delays[axis] is not None else self.delay
            if delay not in intervals:
                intervals[delay] = snapshot.delta(start-delay, end-delay)
            delta = intervals[delay]
            result.append(float(delta[axis]*scale) if scale is not None and delta is not None else 0.)
        return tuple(result)

    def observe(self, snapshot, start, end, shift):
        if snapshot is None:
            return
        if snapshot.generation != self.generation or not snapshot.active:
            self.reset()
            self.generation = snapshot.generation
        if end-self._input_status_at >= .25:
            recent = snapshot.samples[-64:]
            self._input_status = dict(active=snapshot.active, available=snapshot.available,
                history_samples=len(snapshot.samples), latest_batch=recent[-1][0] if recent else None,
                max_gap_ms=max((b[0]-a[0] for a, b in zip(recent, recent[1:])), default=0.)*1000,
                moving_intervals=sum(a[1:] != b[1:] for a, b in zip(recent, recent[1:])))
            self._input_status_at = end
        if not snapshot.active or shift is None or not .001 <= end-start <= .12:
            return
        if not all(math.isfinite(v) for v in (*shift, start, end)):
            return
        invalid = []
        intervals = {}
        for axis, scale in enumerate(self.scales):
            if scale is None:
                continue
            delay = self.delays[axis] if self.delays[axis] is not None else self.delay
            if delay not in intervals:
                intervals[delay] = snapshot.delta(start-delay, end-delay)
            delta = intervals[delay]
            if delta is not None and abs(delta[axis]) >= 2 and abs(delta[axis]*scale-shift[axis]) > max(3., abs(shift[axis])*.25):
                invalid.append(axis)
        if invalid:
            self.scales = tuple(None if axis in invalid else value for axis, value in enumerate(self.scales))
            self.delays = tuple(None if axis in invalid else value for axis, value in enumerate(self.delays))
            self._samples.clear()
            self._fitted_at = -math.inf
            self.delay = float(np.mean([v for v in self.delays if v is not None])) if any(v is not None for v in self.delays) else 0.
            return
        self._samples.append((start, end, *shift))
        if len(self._samples) >= 20 and end-self._fitted_at >= .2:
            self._fit(snapshot, end)

    def _fit(self, snapshot, timestamp):
        self._fitted_at = timestamp
        history = np.asarray(snapshot.samples, dtype=np.float64)
        observations = np.asarray(self._samples, dtype=np.float64)
        if len(history) < 2 or observations[-1, 1]-observations[0, 0] < .3:
            return
        delays = np.arange(0., .0801, .004)
        starts, ends = observations[None, :, 0]-delays[:, None], observations[None, :, 1]-delays[:, None]
        valid = (starts >= history[0, 0]) & (ends <= history[-1, 0]+.008)
        for i in np.flatnonzero(np.diff(history[:, 0]) > .02):
            valid &= ~((starts < history[i+1, 0]) & (ends > history[i, 0]))
        counts = np.stack([np.interp(ends, history[:, 0], history[:, axis])
                           - np.interp(starts, history[:, 0], history[:, axis]) for axis in (1, 2)], axis=2)
        selected = valid[:, :, None] & (np.abs(counts) >= 2)
        lengths = np.count_nonzero(selected, axis=1)
        rows, axes = np.nonzero(lengths >= 20)
        best = [None, None]
        if len(rows):
            x, mask = counts[rows, :, axes], selected[rows, :, axes]
            excited = []
            for row, selection in zip(x, mask):
                compact = row[selection]
                excited.append(not (np.std(compact) < max(1., abs(float(np.mean(compact)))*.12)))
            excited = np.asarray(excited)
            rows, axes, x, mask = rows[excited], axes[excited], x[excited], mask[excited]
            if len(rows):
                sizes = lengths[rows, axes]
                y = observations[:, 2:][:, axes].T
                ratios = np.divide(y, x, out=np.full_like(x, np.inf), where=mask)
                scales = self._quantile_rows(ratios, sizes, .5)
                residual = np.where(mask, np.abs(y-scales[:, None]*x), np.inf)
                errors = self._quantile_rows(residual, sizes, .75)
                truth_medians = self._quantile_rows(np.where(mask, np.abs(y), np.inf), sizes, .5)
                accepted = ((np.abs(scales) >= .005) & (np.abs(scales) <= 20)
                            & (errors <= np.maximum(.8, truth_medians*.12)))
                for axis in (0, 1):
                    candidates = np.flatnonzero(accepted & (axes == axis))
                    if len(candidates):
                        winner = min(candidates, key=lambda i: (errors[i], delays[rows[i]]))
                        best[axis] = (float(errors[winner]), float(delays[rows[winner]]), float(scales[winner]))
        scales, delays = list(self.scales), list(self.delays)
        for axis, candidate in enumerate(best):
            if candidate is not None:
                delays[axis], scales[axis] = candidate[1:]
        self.scales, self.delays = tuple(scales), tuple(delays)
        self.delay = float(np.mean([v for v in delays if v is not None])) if any(v is not None for v in delays) else 0.

    @staticmethod
    def _quantile_rows(values, lengths, quantile):
        ordered = np.sort(values, axis=1)
        ranks = (lengths-1)*quantile
        lower, upper = np.floor(ranks).astype(int), np.ceil(ranks).astype(int)
        rows = np.arange(len(values))
        a, b = ordered[rows, lower], ordered[rows, upper]
        if quantile == .5:
            return (a+b)*.5
        fraction = ranks-lower
        return np.where(fraction >= .5, b-(b-a)*(1-fraction), a+(b-a)*fraction)
