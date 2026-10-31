import math


def _camera_shift(metadata, snapshot, timestamp, duration):
    if not metadata.get('camera_valid'):
        return 0., 0.
    velocity = metadata.get('camera_velocity', (0., 0.))
    camera = tuple(v*duration for v in velocity) if all(math.isfinite(v) for v in velocity) else (0., 0.)
    if snapshot is None:
        return camera
    calibration = metadata.get('calibration') or {}
    scales = calibration.get('scales', (None, None))
    delays = calibration.get('delays') or (calibration.get('delay', 0.),)*2
    values, intervals = [], {}
    for axis, scale in enumerate(scales):
        if scale is None or not math.isfinite(scale):
            values.append(camera[axis])
            continue
        delay = delays[axis] if delays[axis] is not None else calibration.get('delay', 0.)
        if delay not in intervals:
            intervals[delay] = snapshot.delta(timestamp-delay, timestamp+duration-delay)
        delta = intervals[delay]
        values.append(delta[axis]*scale if delta is not None else camera[axis])
    return tuple(values)


def project_detections(detections, metadata, now, history=None):
    if not metadata:
        return detections
    timestamp = metadata.get('timestamp')
    if timestamp is None or not math.isfinite(timestamp) or not math.isfinite(now):
        return []
    age = now-timestamp
    if age < -.001 or age > .1:
        return []
    tracks = metadata.get('tracks', ())
    if len(tracks) != len(detections):
        return []
    width, height = metadata['display_size']
    if width <= 0 or height <= 0:
        return []
    calibration = metadata.get('calibration') or {}
    snapshot = history.read() if history is not None else None
    if snapshot is not None and not snapshot.available:
        snapshot = None
    if snapshot is not None and (not snapshot.active or snapshot.generation != calibration.get('generation')):
        return []
    camera_valid = metadata.get('camera_valid')
    camera_cache = {}
    result = []
    left, top, cw, ch = metadata.get('region', (0, 0, width, height))
    for detection, track in zip(detections, tracks):
        source_time, visual_time = track.get('source_time'), track.get('visual_time')
        if (source_time is None or visual_time is None
                or not all(math.isfinite(v) for v in (source_time, visual_time))
                or not -.001 <= visual_time-source_time <= .1
                or not -.001 <= now-visual_time <= .1):
            continue
        match_quality, samples = track.get('quality', 0.), track.get('motion_samples', 0)
        quality = max(0., min(1., (match_quality-.74)/.11)) if math.isfinite(match_quality) else 0.
        evidence = max(0., min(1., (samples-1)/3)) if math.isfinite(samples) else 0.
        confidence = quality*evidence if track.get('ready') else 0.
        duration = min(.02+.04*confidence, max(0., now-visual_time))
        key = (visual_time, duration)
        if key not in camera_cache:
            camera_cache[key] = _camera_shift(metadata, snapshot, visual_time, duration)
        residual, current_camera = (0., 0.), camera_cache[key]
        if 'screen_velocity' in track and not track.get('ready'):
            current_camera = (0., 0.)
        velocity = track.get('velocity', (0., 0.)) if camera_valid else track.get('screen_velocity', track.get('velocity', (0., 0.)))
        if track.get('ready') and all(math.isfinite(v) for v in velocity):
            residual = tuple(v*duration for v in velocity)
        base_limit = max(0., min(8., (detection[2]-detection[0])*width*.5))
        limit = base_limit+(32.-base_limit)*confidence
        camera_size, target_size = math.hypot(*current_camera), math.hypot(*residual)
        camera_limit = 32.+32.*confidence
        scale = min(1., camera_limit/camera_size if camera_size else 1., limit/target_size if target_size else 1.)
        dx, dy = (current_camera[0]+residual[0])*scale, (current_camera[1]+residual[1])*scale
        box = (detection[0]+dx/width, detection[1]+dy/height,
               detection[2]+dx/width, detection[3]+dy/height, *detection[4:])
        cx, cy = (box[0]+box[2])*width*.5, (box[1]+box[3])*height*.5
        if left <= cx <= left+cw and top <= cy <= top+ch:
            result.append(box)
        else:
            cx, cy = (detection[0]+detection[2])*width*.5, (detection[1]+detection[3])*height*.5
            if left <= cx <= left+cw and top <= cy <= top+ch:
                result.append(detection)
    return result
