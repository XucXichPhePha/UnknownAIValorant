from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from collections.abc import Sequence
from itertools import islice
import math

import cv2
import numpy as np

from motion_tracking import _assign


@dataclass
class _Track:
    identity: int
    box: list
    name: str
    label: str
    last_seen: float
    template: np.ndarray | None = None
    offset: tuple | None = None
    score: float = 1.
    last_patch: np.ndarray | None = None
    last_position: tuple | None = None
    displacement: float = 0.
    mask: np.ndarray | None = None
    velocity: tuple = (0., 0.)
    screen_velocity: tuple = (0., 0.)
    visual_time: float | None = None
    motion_samples: int = 0
    motion_ready: bool = False
    ambiguous: bool = False


class PixelTracker:
    """Track at most ``max_targets`` local patches for ``max_missing`` seconds.

    Input and output boxes are normalized to the supplied cropped frame. A model
    result must carry its source image/time when it predates the current image.
    Missing/ambiguous visual correspondence hides a marker immediately. A good
    visual correspondence bridges model misses for at most 100ms by default.
    Call ``reset`` when capture region or model changes without changing shape.
    """
    def __init__(self, max_missing=.1, max_targets=24, *, assistance=False, mouse_history=None):
        self.max_missing = max(0., min(float(max_missing), .25))
        self.max_targets = max(1, min(int(max_targets), 64))
        self.assistance = assistance
        self.mouse_history = mouse_history
        self._camera = self._calibration = None
        if assistance:
            from camera_motion import CameraMotion, MouseCalibration
            self._camera, self._calibration = CameraMotion(), MouseCalibration()
        self.reset()

    def reset(self):
        self._tracks = []
        self._shape = None
        self._next_id = 1
        self._timestamp = None
        self._thumbnail = None
        self._last_source_time = -math.inf
        self._minimum_source_time = -math.inf
        self._previous_gray = None
        self._camera_estimate = None
        self._camera_velocity = (0., 0.)
        self._camera_history = deque(maxlen=32)
        self._camera_segment = 0
        self._verified_history = deque(maxlen=32)
        if self._calibration is not None:
            self._calibration.reset()

    @property
    def motion_state(self):
        if not self.assistance:
            return None
        return dict(timestamp=self._timestamp, camera_valid=self._camera_estimate is not None,
                    camera_velocity=self._camera_velocity,
                    calibration=self._calibration.state,
                    tracks=[dict(identity=t.identity, velocity=t.velocity, screen_velocity=t.screen_velocity,
                                 visual_time=t.visual_time, quality=t.score, motion_samples=t.motion_samples,
                                 source_time=t.last_seen, ready=t.motion_ready) for t in self._tracks])

    def _camera_between(self, source_time):
        if not self._camera_history:
            return None
        latest = self._camera_history[-1]
        for previous in reversed(self._camera_history):
            if abs(previous[0]-source_time) <= 1e-6 and previous[3] == latest[3]:
                from camera_motion import CameraEstimate
                return CameraEstimate((latest[1]-previous[1], latest[2]-previous[2]), .5)
        return None

    @property
    def active_tracks(self):
        return tuple((track.identity, self._normalized(track)) for track in self._tracks)

    def _normalized(self, track):
        height, width = self._shape
        x1, y1, x2, y2 = track.box
        return (max(0., min(1., x1/width)), max(0., min(1., y1/height)),
                max(0., min(1., x2/width)), max(0., min(1., y2/height)),
                track.name, track.label)

    @staticmethod
    def _gray(frame):
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4
                            else cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _patch(gray, box, head=False):
        """Keep body interiors and complete small head silhouettes.

        A tiny head's interior alone can be only 5px wide. Its outline carries
        the correspondence when face pixels animate or neighboring heads look
        alike. Larger textured heads and bodies use an interior to exclude
        independently animated outlines, limbs and adjacent targets.
        """
        height, width = gray.shape
        w, h = box[2]-box[0], box[3]-box[1]
        pw, ph = (min(20, math.ceil(w)), min(24, math.ceil(h))) if head and min(w, h) < 12 else (
            min(20, int(w*.8)), min(24, int(h*.8)))
        cx, cy = (box[0]+box[2])*.5, (box[1]+box[3])*.5
        x, y = round(cx-pw/2), round(cy-ph/2)
        inside = x >= 0 and y >= 0 and x+pw <= width and y+ph <= height
        if inside and min(pw, ph) >= 5:
            patch = gray[y:y+ph, x:x+pw].copy()
            if float(cv2.meanStdDev(patch)[1][0, 0]) >= 5:
                return patch, (float(x-box[0]), float(y-box[1])), None
        if not head:
            return None, None, None
        # Tiny/flat heads need their edges, but static background must not
        # dominate correspondence when a bot moves independently of the camera.
        pw, ph = min(20, max(8, math.ceil(w)+4)), min(24, max(8, math.ceil(h)+4))
        x, y = round(cx-pw/2), round(cy-ph/2)
        if x < 0 or y < 0 or x+pw > width or y+ph > height:
            return None, None, None
        patch = gray[y:y+ph, x:x+pw].copy()
        if float(cv2.meanStdDev(patch)[1][0, 0]) < 5:
            return None, None, None
        mask = np.full((ph, pw), .15, dtype=np.float32)
        left, top = max(0, round(box[0]-x)), max(0, round(box[1]-y))
        right, bottom = min(pw, round(box[2]-x)), min(ph, round(box[3]-y))
        if right <= left or bottom <= top:
            return None, None, None
        mask[top:bottom, left:right] = 1.
        return patch, (float(x-box[0]), float(y-box[1])), mask

    @staticmethod
    def _locate(gray, track, radius, hint=(0., 0.), adaptive=False, stationary=True):
        if PixelTracker._search(gray, track, radius, hint, stationary):
            return True
        if adaptive and track.ambiguous:
            return False
        # A fast pan gets one bounded wider search; ordinary 120Hz motion uses
        # the small window. Failed/ambiguous pixels still never free-coast.
        fallback = max(64, radius) if adaptive else 24
        return bool(track.template is not None and (radius < fallback or any(hint))
                    and PixelTracker._search(gray, track, fallback, stationary=stationary))

    @staticmethod
    def _head_body_pairs(tracks):
        heads = [t for t in tracks if t.name.casefold() == "head"]
        bodies = [t for t in tracks if t.name.casefold() == "body"]
        candidates = []
        for body in bodies:
            x1, y1, x2, y2 = body.box
            for head in heads:
                hx, hy = (head.box[0]+head.box[2])*.5, (head.box[1]+head.box[3])*.5
                if x1 <= hx <= x2 and y1 <= hy <= y1+(y2-y1)*.35:
                    cost = abs(hx-(x1+x2)*.5)/(x2-x1)+(hy-y1)/(y2-y1)
                    candidates.append((cost, id(body), id(head), head))
        pairs, used_heads = {}, set()
        for _, body_id, head_id, head in sorted(candidates, key=lambda p: p[0]):
            if body_id not in pairs and head_id not in used_heads:
                pairs[body_id] = head
                used_heads.add(head_id)
        return pairs

    @staticmethod
    def _verify_shift(gray, track, dx, dy, radius=24):
        """Accept a head hint only when exact body pixels have a unique place.

        A loose correlation threshold can accept a shifted smooth shirt. Exact
        native pixel agreement is deliberately conservative; animation, model
        jitter and independent body movement fall back to the normal search.
        """
        if track.template is None:
            return False
        ph, pw = track.template.shape
        if track.last_patch is not None:
            lx, ly = track.last_position
            if cv2.norm(gray[ly:ly+ph, lx:lx+pw], track.last_patch, cv2.NORM_INF) == 0:
                # A striped shirt may fit the moving head's hint even while
                # its own measured pixels stopped. Preserve the stop first.
                return False
        ex, ey = track.box[0]+track.offset[0], track.box[1]+track.offset[1]
        x, y = round(ex+dx), round(ey+dy)
        if x < 0 or y < 0 or x+pw > gray.shape[1] or y+ph > gray.shape[0]:
            return False
        patch = gray[y:y+ph, x:x+pw]
        if cv2.norm(patch, track.template, cv2.NORM_INF) != 0:
            return False
        x1, y1 = max(0, math.floor(ex-radius)), max(0, math.floor(ey-radius))
        x2, y2 = min(gray.shape[1], math.ceil(ex+pw+radius)), min(gray.shape[0], math.ceil(ey+ph+radius))
        if x < x1 or y < y1 or x+pw > x2 or y+ph > y2:
            return False
        errors = cv2.matchTemplate(gray[y1:y2, x1:x2], track.template, cv2.TM_SQDIFF)
        px, py = x-x1, y-y1
        suppression = max(2, min(pw, ph)//5)
        errors[max(0, py-suppression):py+suppression+1,
               max(0, px-suppression):px+suppression+1] = math.inf
        if cv2.minMaxLoc(errors)[0]/(pw*ph) < 9:
            return False
        sx, sy = x-ex, y-ey
        track.box[0] += sx
        track.box[2] += sx
        track.box[1] += sy
        track.box[3] += sy
        track.last_patch = patch.copy()
        track.last_position = (x, y)
        track.displacement, track.score = math.hypot(sx, sy), 1.
        return True

    def _advance(self, gray, timestamp, radius, camera=None):
        eligible = [t for t in self._tracks if timestamp-t.last_seen <= self.max_missing]
        pairs = self._head_body_pairs(eligible)
        moved, retained = {}, set()
        for track in sorted(eligible, key=lambda t: t.name.casefold() == "body"):
            before_x, before_y = track.box[:2]
            head = pairs.get(id(track))
            verified = (head is not None and id(head) in moved
                        and self._verify_shift(gray, track, *moved[id(head)], radius=max(24, radius)))
            interval = 0. if track.visual_time is None else max(0., timestamp-track.visual_time)
            hint = (0., 0.)
            if self.assistance:
                velocity = track.velocity if camera is not None else track.screen_velocity
                hint = tuple(max(-96., min(96., (camera[axis] if camera is not None else 0.)
                                                + velocity[axis]*interval)) for axis in (0, 1))
            stationary = camera is None or math.hypot(*camera) < 1.
            if verified or self._locate(gray, track, radius, hint, self.assistance, stationary):
                moved[id(track)] = (track.box[0]-before_x, track.box[1]-before_y)
                if self.assistance:
                    self._measure_motion(track, moved[id(track)], camera, interval, timestamp)
                    if camera is not None:
                        track.displacement = math.hypot(*(v-c for v, c in zip(moved[id(track)], camera)))
                retained.add(id(track))
        self._tracks = [t for t in eligible if id(t) in retained]

    @staticmethod
    def _measure_motion(track, shift, camera, interval, timestamp):
        track.visual_time = timestamp
        if not .001 <= interval <= .08:
            track.velocity = track.screen_velocity = (0., 0.)
            track.motion_samples = 0
            track.motion_ready = False
            return
        previous = track.screen_velocity
        screen = tuple(v/interval for v in shift)
        track.screen_velocity = screen if math.hypot(*shift) > .3 else (0., 0.)
        track.velocity = tuple((v-c)/interval for v, c in zip(shift, camera)) if camera is not None else (0., 0.)
        before, speed = math.hypot(*previous), math.hypot(*screen)
        direction = (before > 0 and speed > 0
                     and sum(a*b for a, b in zip(previous, screen)) > .5*before*speed)
        if not any(track.screen_velocity) or not math.isfinite(track.score) or track.score < .74:
            track.motion_samples = 0
        else:
            track.motion_samples = track.motion_samples+1 if direction else 1
        track.motion_ready = track.motion_samples >= 2

    @staticmethod
    def _search(gray, track, radius, hint=(0., 0.), stationary=True):
        track.ambiguous = False
        template = track.template
        if template is None:
            return False
        height, width = gray.shape
        ph, pw = template.shape
        if track.last_patch is not None and stationary:
            lx, ly = track.last_position
            # Repeated capture pixels keep their exact last marker coordinates
            # and need no search. This also saves work for stationary targets.
            if cv2.norm(gray[ly:ly+ph, lx:lx+pw], track.last_patch, cv2.NORM_INF) == 0:
                track.displacement = 0.
                return True
        ex, ey = track.box[0]+track.offset[0], track.box[1]+track.offset[1]
        hx, hy = ex+hint[0], ey+hint[1]
        x1, y1 = max(0, math.floor(hx-radius)), max(0, math.floor(hy-radius))
        x2, y2 = min(width, math.ceil(hx+pw+radius)), min(height, math.ceil(hy+ph+radius))
        region = gray[y1:y2, x1:x2]
        if region.shape[0] < ph or region.shape[1] < pw:
            return False
        if track.mask is not None:
            weight = float(cv2.norm(track.mask, cv2.NORM_L2SQR))
            scores = -cv2.matchTemplate(region, template, cv2.TM_SQDIFF,
                                         mask=track.mask)/weight
            _, peak, _, (x, y) = cv2.minMaxLoc(scores)
            if not math.isfinite(peak) or peak < -64:
                return False
            quality, ambiguity = max(.75, 1+peak/256), 9.
        else:
            scores = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
            _, peak, _, (x, y) = cv2.minMaxLoc(scores)
            if peak < .74:
                return False
            quality, ambiguity = peak, .06
        patch = region[y:y+ph, x:x+pw]
        # Correlation alone accepts severe contrast/lighting changes and some
        # occlusions. Require the actual pixels to remain reasonably similar.
        if track.mask is None and cv2.norm(patch, template, cv2.NORM_L1)/(pw*ph) > 28:
            return False
        px, py = float(x), float(y)
        if 0 < x < scores.shape[1]-1:
            a, b, c = map(float, scores[y, x-1:x+2])
            denominator = a-2*b+c
            if denominator < -.001:
                px += max(-.5, min(.5, .5*(a-c)/denominator))
        if 0 < y < scores.shape[0]-1:
            a, b, c = map(float, scores[y-1:y+2, x])
            denominator = a-2*b+c
            if denominator < -.001:
                py += max(-.5, min(.5, .5*(a-c)/denominator))
        # An equally plausible second place can be another bot. Hide instead
        # of allowing two old markers to claim the same target's pixels.
        suppression = max(2, min(pw, ph)//5)
        scores[max(0, y-suppression):y+suppression+1,
               max(0, x-suppression):x+suppression+1] = -math.inf
        if cv2.minMaxLoc(scores)[1] > peak-ambiguity:
            track.ambiguous = True
            return False
        dx, dy = x1+px-ex, y1+py-ey
        track.box[0] += dx
        track.box[2] += dx
        track.box[1] += dy
        track.box[3] += dy
        track.score = quality
        track.displacement = math.hypot(dx, dy)
        track.last_patch = patch.copy()
        track.last_position = (x1+x, y1+y)
        return True

    def _observations(self, detections, source, timestamp, current, radius):
        height, width = current.shape
        proposals = []
        # Model outputs are materialized and capped at300. Inspect those before
        # budgeting patches, so Body-first output order cannot suppress Heads.
        # Unbounded iterators keep a separate strict consumption limit.
        materialized = isinstance(detections, (Sequence, np.ndarray))
        budget = 300 if materialized else self.max_targets*4
        raw = list(islice(detections, budget))
        raw.sort(key=lambda d: len(d) < 5 or str(d[4]).casefold() != "head")
        for detection in raw:
            if len(proposals) >= self.max_targets*2:
                break
            if len(detection) < 6:
                continue
            box = [float(detection[0])*width, float(detection[1])*height,
                   float(detection[2])*width, float(detection[3])*height]
            if (not all(map(math.isfinite, box)) or box[2] <= box[0] or box[3] <= box[1]):
                continue
            template, offset, mask = self._patch(source, box,
                                                str(detection[4]).casefold() == "head")
            track = _Track(0, box.copy(), str(detection[4]), str(detection[5]), timestamp,
                           template, offset)
            track.mask = mask
            proposals.append(track)
        proposals.sort(key=lambda t: t.name.casefold() != "head")
        camera = self._camera_between(timestamp) if radius and self.assistance else None
        if radius and self.assistance and camera is None:
            mouse_hint = self._calibration.project(self.mouse_history.read(), timestamp, self._timestamp) if self.mouse_history is not None else None
            camera = self._camera.estimate(source, current, [t.box for t in proposals], hint=mouse_hint)
        hint = camera.shift if camera is not None else (0., 0.)
        pairs = self._head_body_pairs(proposals)
        moved, observations = {}, []
        for track in proposals:
            if len(observations) >= self.max_targets:
                break
            box, template, offset = track.box.copy(), track.template, track.offset
            head = pairs.get(id(track))
            body_verified = (radius and head is not None and id(head) in moved
                             and self._verify_shift(current, track, *moved[id(head)], radius=max(24, radius)))
            if radius and not body_verified:
                # Nearest current markers cannot disambiguate the old source
                # identity: a6px hint window may exclude an equally plausible
                # neighboring head. Compare the entire bounded source window.
                if not self._locate(current, track, radius, hint, self.assistance):
                    continue
            if not radius and template is not None:
                track.last_patch = template
                track.last_position = (round(box[0]+offset[0]), round(box[1]+offset[1]))
            observations.append(track)
            track.visual_time = self._timestamp
            moved[id(track)] = (track.box[0]-box[0], track.box[1]-box[1])
            if camera is not None:
                track.displacement = math.hypot(*(v-c for v, c in zip(moved[id(track)], camera.shift)))
        return observations

    def _renew_from_source(self, detections, source, timestamp):
        historical = next((tracks for captured, tracks in reversed(self._verified_history)
                           if abs(captured-timestamp) <= 1e-6), ())
        if not historical or not self._tracks or not detections:
            return
        height, width = source.shape
        valid = []
        for detection in detections:
            if len(detection) < 6:
                continue
            box = [float(detection[0])*width, float(detection[1])*height,
                   float(detection[2])*width, float(detection[3])*height]
            if all(map(math.isfinite, box)) and box[2] > box[0] and box[3] > box[1]:
                valid.append((box, str(detection[4]).casefold(), str(detection[5])))
        if not valid:
            return
        before = np.asarray([item[1] for item in historical])
        after = np.asarray([item[0] for item in valid])
        sizes, other_sizes = before[:, 2:]-before[:, :2], after[:, 2:]-after[:, :2]
        distances = np.linalg.norm((before[:, None, :2]+before[:, None, 2:]
                                   -after[None, :, :2]-after[None, :, 2:])*.5, axis=2)
        gate = np.minimum(sizes.min(axis=1)[:, None], other_sizes.min(axis=1)[None])*.35
        ratios = np.maximum(sizes[:, None]/other_sizes[None], other_sizes[None]/sizes[:, None]).max(axis=2)
        intersection = np.maximum(0., np.minimum(before[:, None, 2:], after[None, :, 2:])
                                   -np.maximum(before[:, None, :2], after[None, :, :2])).prod(axis=2)
        union = sizes.prod(axis=1)[:, None]+other_sizes.prod(axis=1)[None]-intersection
        names = np.asarray([item[2].casefold() for item in historical])
        other_names = np.asarray([item[1] for item in valid])
        matches = ((names[:, None] == other_names[None]) & (distances <= gate)
                   & (ratios <= 2.) & (intersection >= union*.45))
        unique_rows, unique_columns = matches.sum(axis=1) == 1, matches.sum(axis=0) == 1
        current = {track.identity: track for track in self._tracks}
        for row, column in zip(*np.nonzero(matches & unique_rows[:, None] & unique_columns[None])):
            identity, _, _, patch, position = historical[row]
            track = current.get(identity)
            if track is None or patch is None or position is None:
                continue
            x, y = position
            ph, pw = patch.shape
            actual = source[y:y+ph, x:x+pw]
            if actual.shape == patch.shape and cv2.norm(actual, patch, cv2.NORM_INF) == 0:
                track.last_seen = max(track.last_seen, timestamp)
                track.label = valid[column][2]

    def _correct(self, observations):
        costs = np.full((len(self._tracks), len(observations)), 1e6)
        if self._tracks and observations:
            before = np.asarray([t.box for t in self._tracks])
            after = np.asarray([t.box for t in observations])
            sizes, other_sizes = before[:, 2:]-before[:, :2], after[:, 2:]-after[:, :2]
            ratios = np.maximum(sizes[:, None]/other_sizes[None],
                                other_sizes[None]/sizes[:, None]).max(axis=2)
            centers = (before[:, :2]+before[:, 2:])*.5
            other_centers = (after[:, :2]+after[:, 2:])*.5
            distance = np.linalg.norm(centers[:, None]-other_centers[None], axis=2)
            gates = np.clip(sizes[:, 0]*.8, 8, 24)
            costs = distance/gates[:, None]+.15*np.log(ratios)
            names = np.array([t.name.casefold() for t in self._tracks])
            other_names = np.array([t.name.casefold() for t in observations])
            # Small heads often change detector size by more than2x while
            # keeping the same center. Treat that as a correction instead of
            # retaining the old box beside a new identity for another100ms.
            near_center = distance <= np.minimum(sizes.min(axis=1)[:, None],
                                                  other_sizes.min(axis=1)[None])*.5
            costs[(names[:, None] != other_names[None])
                  | ((ratios > 2) & ~near_center)] = 1e6
        matches = dict(_assign(costs))
        used = set(matches.values())
        retained = []
        for index, track in enumerate(self._tracks):
            if index in matches:
                replacement = observations[matches[index]]
                replacement.identity = track.identity
                if self.assistance:
                    delta = math.hypot((replacement.box[0]+replacement.box[2]-track.box[0]-track.box[2])*.5,
                                       (replacement.box[1]+replacement.box[3]-track.box[1]-track.box[3])*.5)
                    if delta <= max(1., min(track.box[2]-track.box[0], track.box[3]-track.box[1])*.3):
                        replacement.velocity = track.velocity
                        replacement.screen_velocity = track.screen_velocity
                        replacement.motion_samples = track.motion_samples
                        replacement.motion_ready = track.motion_ready
                retained.append(replacement)
            else:
                retained.append(track)
        for index, track in enumerate(observations):
            if index not in used:
                track.identity = self._next_id
                self._next_id += 1
                retained.append(track)
        retained.sort(key=lambda t: t.name.casefold() != "head")
        self._tracks = retained[:self.max_targets]

    def _unique_correspondence(self):
        """One visible patch cannot carry two same-class marker identities."""
        if len(self._tracks) < 2:
            return
        chosen = []
        for index in sorted(range(len(self._tracks)), key=lambda i:
                self._tracks[i].score-.002*self._tracks[i].displacement, reverse=True):
            track = self._tracks[index]
            x1, y1, x2, y2 = track.box
            cx, cy = (x1+x2)*.5, (y1+y2)*.5
            duplicate = False
            for other_index in chosen:
                other = self._tracks[other_index]
                if track.name.casefold() != other.name.casefold():
                    continue
                ox1, oy1, ox2, oy2 = other.box
                gate = max(2., min(x2-x1, ox2-ox1)*.18)
                dx, dy = cx-(ox1+ox2)*.5, cy-(oy1+oy2)*.5
                if dx*dx+dy*dy < gate*gate:
                    duplicate = True
                    break
            if not duplicate:
                chosen.append(index)
        self._tracks = [self._tracks[index] for index in sorted(chosen)]

    def update(self, frame, timestamp, *, detections=None, source_frame=None, source_time=None):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite monotonic seconds")
        gray = self._gray(frame)
        shape = gray.shape
        thumbnail = gray[::10, ::10]
        dt = None if self._timestamp is None else timestamp-self._timestamp
        snapshot = self.mouse_history.read() if self.assistance and self.mouse_history is not None else None
        estimate = None
        if self.assistance and self._previous_gray is not None and dt is not None and 0 < dt <= .12:
            mouse_hint = self._calibration.project(snapshot, self._timestamp, timestamp) if snapshot is not None else None
            estimate = self._camera.estimate(self._previous_gray, gray, [t.box for t in self._tracks], hint=mouse_hint)
        reset = (self._shape is not None and shape != self._shape) or (
            dt is not None and (dt < 0 or dt > .25))
        if (self._thumbnail is not None and thumbnail.shape == self._thumbnail.shape
                and cv2.norm(thumbnail, self._thumbnail, cv2.NORM_L1)/thumbnail.size > 45
                and estimate is None):
            reset = True
        if reset:
            self._tracks = []
            self._minimum_source_time = timestamp
            self._camera_history.clear()
            self._verified_history.clear()
            if self._calibration is not None:
                self._calibration.reset()
            estimate = None
        if self.assistance and self.mouse_history is not None:
            previous_generation = self._calibration.generation
            self._calibration.observe(snapshot, timestamp-(dt or 0.), timestamp,
                                      estimate.shift if estimate is not None else None)
            if snapshot is not None and previous_generation is not None and snapshot.generation != previous_generation:
                self._tracks = []
                self._minimum_source_time = timestamp
                self._camera_history.clear()
                self._verified_history.clear()
                estimate = None
        if self.assistance:
            if estimate is None or not self._camera_history:
                self._camera_segment += 1
                position = (0., 0.)
            else:
                position = tuple(a+b for a, b in zip(self._camera_history[-1][1:3], estimate.shift))
            self._camera_history.append((timestamp, *position, self._camera_segment))
        self._previous_gray = gray if self.assistance else None
        self._camera_estimate = estimate
        self._camera_velocity = tuple(v/dt for v in estimate.shift) if estimate is not None and dt is not None and .001 <= dt <= .08 else (0., 0.)
        self._shape, self._timestamp, self._thumbnail = shape, timestamp, thumbnail.copy()
        source = None
        observed_time = timestamp if detections is None or source_time is None else float(source_time)
        age = timestamp-observed_time
        if (detections is not None and math.isfinite(observed_time)
                and observed_time > self._last_source_time and observed_time >= self._minimum_source_time
                and -.001 <= age <= self.max_missing):
            self._last_source_time = observed_time
            candidate = gray if source_frame is None or source_frame is frame else self._gray(source_frame)
            if candidate.shape == shape and (source_frame is not None or age <= 1e-9):
                source = candidate
                materialized = isinstance(detections, (Sequence, np.ndarray))
                detections = list(islice(detections, 300 if materialized else self.max_targets*4))
                self._renew_from_source(detections, source, observed_time)
        radius = min(32, max(8, math.ceil(max(dt or 0, 0)*900)))
        self._advance(gray, timestamp, radius, estimate.shift if estimate is not None else None)
        self._unique_correspondence()
        if source is not None:
            delayed = age > 1e-9 and source_frame is not frame
            observations = self._observations(detections, source, observed_time, gray,
                min(64, max(16, math.ceil(age*1600))) if delayed else 0)
            self._correct(observations)
            self._unique_correspondence()
        self._verified_history.append((timestamp, tuple((t.identity, tuple(t.box), t.name,
                                                        t.last_patch, t.last_position) for t in self._tracks)))
        return [self._normalized(track) for track in self._tracks]
