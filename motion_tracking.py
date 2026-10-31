"""Short-lived screen-space tracks, compensated for background camera motion.

No additional model pass or runtime dependency. Observations always take priority;
only confirmed tracks bridge up to 100ms of missed detections.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import cv2
import numpy as np


@dataclass
class CameraMotion:
    matrix: np.ndarray = field(default_factory=lambda: np.eye(2, 3))
    reliable: bool = False
    scene_cut: bool = False


class CameraMotionEstimator:
    """Sparse background flow on a 160px thumbnail; targets/HUD/weapon are masked."""
    def __init__(self):
        self._previous = None
        self._shape = None

    def update(self, frame, previous_detections=()):
        height, width = frame.shape[:2]
        small = cv2.resize(frame, (160, max(32, round(height * 160 / width))))
        gray = cv2.cvtColor(small, cv2.COLOR_BGRA2GRAY if small.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
        previous, shape = self._previous, self._shape
        self._previous, self._shape = gray, frame.shape[:2]
        if previous is None or shape != self._shape:
            return CameraMotion(scene_cut=previous is not None)
        change = float(cv2.absdiff(previous, gray).mean())
        if change < .15:
            return CameraMotion(reliable=np.ptp(gray) > 12)
        sh, sw = gray.shape
        mask = np.full_like(gray, 255)
        mask[:round(sh*.12)] = 0
        mask[round(sh*.78):] = 0
        mask[sh//2:,round(sw*.8):] = 0
        for box in previous_detections:
            x1,y1,x2,y2 = box[:4]
            cv2.rectangle(mask, (max(0,round(x1*sw)-3),max(0,round(y1*sh)-3)),
                          (min(sw-1,round(x2*sw)+3),min(sh-1,round(y2*sh)+3)), 0, -1)
        points = cv2.goodFeaturesToTrack(previous, 48, .015, 6, mask=mask)
        if points is None or len(points) < 8:
            return CameraMotion(scene_cut=change > 45)
        moved, status, error = cv2.calcOpticalFlowPyrLK(previous, gray, points, None,
            winSize=(15,15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,12,.03))
        valid = status.ravel().astype(bool) & (error.ravel() < 25)
        before, after = points[valid,0], moved[valid,0]
        if len(before) < 8:
            return CameraMotion(scene_cut=change > 45)
        affine, inliers = cv2.estimateAffinePartial2D(before, after, method=cv2.RANSAC,
            ransacReprojThreshold=1., maxIters=100, confidence=.95, refineIters=3)
        if affine is None or inliers is None or inliers.sum() < max(8,len(before)*.55):
            return CameraMotion(scene_cut=change > 45)
        support = before[inliers.ravel().astype(bool)]
        scale = math.hypot(affine[0,0], affine[1,0])
        if (np.ptp(support[:,0]) < sw*.25 or np.ptp(support[:,1]) < sh*.2
                or not .9 <= scale <= 1.1 or abs(math.atan2(affine[1,0],affine[0,0])) > .1
                or np.linalg.norm(affine[:,2]) > 30):
            return CameraMotion(scene_cut=change > 45)
        warped = cv2.warpAffine(previous, affine, (sw,sh))
        interior = cv2.warpAffine(mask, affine, (sw,sh)) > 250
        mismatch = float(cv2.absdiff(warped,gray)[interior].mean()) if interior.any() else 255
        if mismatch > 40:
            return CameraMotion(scene_cut=change > 45)
        # Convert thumbnail coordinates back to native screen pixels.
        ratios = np.array([width/sw,height/sh])
        affine[:,:2] = affine[:,:2] * ratios[:,None] / ratios[None,:]
        affine[:,2] *= ratios
        return CameraMotion(affine, reliable=True)


@dataclass
class _Track:
    identity: int
    box: np.ndarray
    name: str
    label: str
    velocity: np.ndarray
    last_seen: float
    hits: int = 1


def _assign(costs):
    """Solve independent gated groups; bound the cost of pathological crowds."""
    valid = costs < 1
    # Most separated targets have just one possible partner. Resolve these in
    # one step instead of constructing a Hungarian problem per isolated bot.
    single = valid & (valid.sum(axis=1)[:,None] == 1) & (valid.sum(axis=0)[None,:] == 1)
    rr,cc = np.where(single)
    result = list(zip(map(int,rr),map(int,cc)))
    valid[rr,:] = False
    remaining = set(np.flatnonzero(valid.any(axis=1)))
    while remaining:
        rows={min(remaining)}
        columns=set()
        while True:
            next_columns=set(np.flatnonzero(valid[sorted(rows)].any(axis=0)))
            next_rows=set(np.flatnonzero(valid[:,sorted(next_columns)].any(axis=1)))
            if next_rows == rows and next_columns == columns: break
            rows,columns=next_rows,next_columns
        remaining -= rows
        rows,columns=sorted(rows),sorted(columns)
        group=costs[np.ix_(rows,columns)]
        if min(group.shape) == 1:
            r,c = np.unravel_index(group.argmin(),group.shape)
            result.append((rows[r],columns[c]))
        elif max(group.shape) <= 24:
            result.extend((rows[r],columns[c]) for r,c in _hungarian(group))
        else:
            # A dense group this large is usually detection clutter. A sorted
            # greedy assignment remains one-to-one without a CPU-time spike.
            rr,cc=np.where(group < 1)
            used_rows,used_columns=set(),set()
            for index in np.argsort(group[rr,cc],kind='stable'):
                r,c=int(rr[index]),int(cc[index])
                if r not in used_rows and c not in used_columns:
                    result.append((rows[r],columns[c]))
                    used_rows.add(r);used_columns.add(c)
    return result


def _hungarian(costs):
    """Minimum-cost one-to-one assignment with unmatched dummy columns.

    Hungarian algorithm, bounded by the small number of visible tracks. Dummy
    cost 1 means invalid or distant observations start their own tracks.
    """
    n, m = costs.shape
    if not n or not m:
        return []
    a = np.concatenate((costs, np.ones((n,n))), axis=1)
    columns = a.shape[1]
    u, v = np.zeros(n+1), np.zeros(columns+1)
    p, way = np.zeros(columns+1,dtype=int), np.zeros(columns+1,dtype=int)
    for i in range(1,n+1):
        p[0], j0 = i, 0
        minimum, used = np.full(columns+1,np.inf), np.zeros(columns+1,dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = np.inf, 0
            for j in range(1,columns+1):
                if used[j]: continue
                current = a[i0-1,j-1] - u[i0] - v[j]
                if current < minimum[j]: minimum[j],way[j] = current,j0
                if minimum[j] < delta: delta,j1 = minimum[j],j
            u[p[used]] += delta
            v[used] -= delta
            minimum[~used] -= delta
            j0 = j1
            if p[j0] == 0: break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return [(p[j]-1,j-1) for j in range(1,m+1) if p[j] and costs[p[j]-1,j-1] < 1]


def _warp_box(box, matrix):
    points = np.array([[box[0],box[1]],[box[2],box[1]],
                       [box[2],box[3]],[box[0],box[3]]]) @ matrix[:,:2].T + matrix[:,2]
    return np.concatenate((points.min(axis=0),points.max(axis=0)))


class MotionTracker:
    def __init__(self, max_missing=.1):
        self.max_missing = max_missing
        self._tracks = []
        self._next_id = 1
        self._timestamp = None
        self._shape = None
        self._camera_reliable = None

    @property
    def active_tracks(self):
        """Identity and latest native observation/prediction, normalized for diagnostics."""
        return tuple((t.identity,self._normalized(t.box,t)) for t in self._tracks)

    def _normalized(self, box, track):
        height,width = self._shape
        b = np.clip(box / [width,height,width,height],0,1)
        return (*map(float,b),track.name,track.label)

    def update(self, detections, timestamp, frame_shape, *, motion=None, display_time=None, region=None):
        # No external camera input means an explicit stationary-camera test/use.
        # The live worker always supplies its estimator, including failures.
        motion = motion if motion is not None else CameraMotion(reliable=True)
        shape = tuple(frame_shape[:2])
        dt = 0 if self._timestamp is None else timestamp-self._timestamp
        if motion.scene_cut or shape != self._shape or not 0 < dt <= .25:
            self._tracks.clear()
            dt = 0
        self._timestamp,self._shape = timestamp,shape
        height,width = shape
        observations = [(np.array(b[:4])*[width,height,width,height],b[4],b[5])
                        for b in detections if len(b) >= 6 and np.isfinite(b[:4]).all()
                        and b[2]>b[0] and b[3]>b[1]][:128]
        if self._camera_reliable != motion.reliable:
            for track in self._tracks: track.velocity[:] = 0
        self._camera_reliable = motion.reliable
        affine = motion.matrix if motion.reliable else np.eye(2,3)
        predicted=[]
        for track in self._tracks:
            track.velocity = affine[:,:2] @ track.velocity
            predicted.append(_warp_box(track.box,affine) + np.tile(track.velocity*dt,2))
        costs = np.full((len(predicted),len(observations)),1e6)
        if predicted and observations:
            # One vectorized cost matrix avoids hundreds of tiny NumPy calls
            # in crowded frames on this CPU.
            predicted_array = np.asarray(predicted)
            observed_array = np.asarray([o[0] for o in observations])
            sizes = predicted_array[:,2:] - predicted_array[:,:2]
            other_sizes = observed_array[:,2:] - observed_array[:,:2]
            ratios = np.maximum(sizes[:,None]/other_sizes[None],
                                other_sizes[None]/sizes[:,None]).max(axis=2)
            centers = (predicted_array[:,:2]+predicted_array[:,2:])/2
            other_centers = (observed_array[:,:2]+observed_array[:,2:])/2
            distances = np.linalg.norm(centers[:,None]-other_centers[None],axis=2)
            gates = np.clip(1.5*sizes[:,0]+300*dt,12,80)
            costs = distances/gates[:,None] + .18*np.log(ratios)
            names = np.array([t.name.casefold() for t in self._tracks])
            other_names = np.array([o[1].casefold() for o in observations])
            costs[(ratios > 2.5) | (names[:,None] != other_names[None])] = 1e6
        assignments=dict(_assign(costs))
        used=set(assignments.values())
        retained=[]
        for i,track in enumerate(self._tracks):
            if i in assignments:
                box,name,label=observations[assignments[i]]
                error=(box[:2]+box[2:]-predicted[i][:2]-predicted[i][2:])/2
                if dt:
                    track.velocity += .45*error/dt
                    speed=np.linalg.norm(track.velocity)
                    if speed>900: track.velocity *= 900/speed
                track.box,track.label,track.last_seen=box,label,timestamp
                track.hits += 1
            else:
                if not motion.reliable or track.hits < 2 or timestamp-track.last_seen > self.max_missing: continue
                track.box=predicted[i]
                track.velocity *= .9
            retained.append(track)
        for j,(box,name,label) in enumerate(observations):
            if j not in used:
                retained.append(_Track(self._next_id,box,name,label,np.zeros(2),timestamp))
                self._next_id += 1
        # Expire targets that leave the actual capture view, including focus view.
        left,top,rw,rh = region or (0,0,width,height)
        self._tracks = [t for t in retained if left <= (t.box[0]+t.box[2])/2 <= left+rw
                        and top <= (t.box[1]+t.box[3])/2 <= top+rh][:128]
        lead=min(.04,max(0,(display_time or timestamp)-timestamp)) if motion.reliable else 0
        result=[]
        for track in self._tracks:
            center=(track.box[:2]+track.box[2:])/2
            camera_velocity=((affine[:,:2] @ center + affine[:,2]-center)/dt) if dt else np.zeros(2)
            offset=(track.velocity+camera_velocity)*lead
            limit=max(6,min(12,(track.box[2]-track.box[0])*.75))
            length=np.linalg.norm(offset)
            if length>limit: offset *= limit/length
            result.append(self._normalized(track.box+np.tile(offset,2),track))
        return result
