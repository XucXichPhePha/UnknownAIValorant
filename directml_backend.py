from __future__ import annotations
import ast
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

@dataclass(frozen=True)
class Letterbox:
    width: int
    height: int
    left: int
    top: int

def validate_capture_size(width: int, height: int) -> None:
    if (width, height) not in ((1280, 882), (1280, 720)):
        raise ValueError(f'Primary display is {width}x{height}. Change its native resolution to 1280x882 (primary target) or 1280x720 before preparing frames.')

def frame_delay(fps: int, elapsed: float) -> float:
    if not 1 <= fps <= 240:
        raise ValueError('FPS must be between 1 and 240')
    return max(0.0, 1.0 / fps - elapsed)

def prepare_frame(frame: np.ndarray, size: int | tuple[int, int]) -> tuple[np.ndarray, Letterbox]:
    height, width = frame.shape[:2]
    target_width, target_height = (size, size) if isinstance(size, int) else size
    scale = min(target_width / width, target_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left, top = ((target_width - resized_width) // 2, (target_height - resized_height) // 2)
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    padded = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
    padded[top:top + resized_height, left:left + resized_width] = resized[:, :, :3]
    tensor = cv2.dnn.blobFromImage(padded, 1 / 255.0, swapRB=True, crop=False)
    return (tensor, Letterbox(resized_width, resized_height, left, top))

def capture_region(width, height, view):
    if view == 'full':
        return (0, 0, width, height)
    sizes = {'focus': (640, 448), 'tight': (512, 384), 'fast': (384, 256), 'compact': (256, 192)}
    if view not in sizes:
        raise ValueError(f'Unsupported detection view: {view}')
    w, h = sizes[view]
    if width < w or height < h:
        raise ValueError(f'{view} view requires a frame of at least {w}x{h}')
    left, top = ((width - w) // 2, (height - h) // 2)
    return (left, top, w, h)

def detection_view(frame, view):
    height, width = frame.shape[:2]
    region = capture_region(width, height, view)
    if view == 'full':
        return (frame, region)
    left, top, w, h = region
    return (frame[top:top + h, left:left + w], region)

def remap_detections(detections, region, frame_shape):
    left, top, width, height = region
    full_height, full_width = frame_shape[:2]
    if region == (0, 0, full_width, full_height):
        return detections
    return [((left + x1 * width) / full_width, (top + y1 * height) / full_height, (left + x2 * width) / full_width, (top + y2 * height) / full_height, name, label) for x1, y1, x2, y2, name, label in detections]

def _plausible_head(box, bodies, width=1, height=1):
    x1, y1, x2, y2 = box[:4]
    head_width, head_height = ((x2 - x1) * width, (y2 - y1) * height)
    if head_height <= 0 or not 0.45 <= head_width / head_height <= 2:
        return False
    cx, cy = ((x1 + x2) / 2, (y1 + y2) / 2)
    nearby = [b for b in bodies if b[0] <= cx <= b[2] and b[1] - 0.15 * (b[3] - b[1]) <= cy <= b[3]]
    return not nearby or any((cy <= b[1] + 0.4 * (b[3] - b[1]) and y2 - y1 <= 0.4 * (b[3] - b[1]) or (y2 - y1 <= 0.18 * (b[3] - b[1]) and x2 - x1 <= 0.5 * (b[2] - b[0])) for b in nearby))

def filter_head_detections(detections, frame_shape):
    height, width = frame_shape[:2]
    bodies = [box for box in detections if box[4].casefold() == 'body']
    return [box for box in detections if box[4].casefold() != 'head' or _plausible_head(box, bodies, width, height)]

def decode_output(output, geometry, names, *, end_to_end=False, confidence=0.25, iou=0.7, head_iou=None, head_geometry=False, return_scores=False, head_confidence=None, quality_filter=False):
    output = np.asarray(output)
    minimum = min(confidence, head_confidence) if head_confidence is not None else confidence
    if output.ndim != 3 or output.shape[0] != 1:
        raise ValueError(f'Unsupported detection output shape: {output.shape}')
    if end_to_end:
        if output.shape[2] != 6:
            raise ValueError(f'Expected [1, N, 6] output; got {output.shape}')
        rows = output[0]
        rows = rows[np.isfinite(rows).all(axis=1) & (rows[:, 4] >= minimum)]
        rows = rows[(rows[:, 5] >= 0) & (rows[:, 5] < len(names)) & (rows[:, 5] == np.floor(rows[:, 5]))]
        boxes, scores, classes = (rows[:, :4].copy(), rows[:, 4], rows[:, 5].astype(int))
    else:
        if output.shape[1] != 4 + len(names):
            raise ValueError(f'Expected [1, {4 + len(names)}, N] output; got {output.shape}')
        rows = output[0].T
        rows = rows[np.isfinite(rows).all(axis=1)]
        classes = rows[:, 4:].argmax(axis=1)
        scores = rows[np.arange(len(rows)), 4 + classes]
        mask = scores >= minimum
        rows, scores, classes = (rows[mask], scores[mask], classes[mask])
        boxes = np.empty((len(rows), 4), dtype=np.float32)
        boxes[:, :2] = rows[:, :2] - rows[:, 2:4] / 2
        boxes[:, 2:] = rows[:, :2] + rows[:, 2:4] / 2
    if head_confidence is not None:
        head_ids = [index for index, name in names.items() if name.casefold() == 'head']
        threshold = np.where(np.isin(classes, head_ids), head_confidence, confidence)
        mask = scores >= threshold
        boxes, scores, classes = (boxes[mask], scores[mask], classes[mask])
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores, classes = (boxes[valid], scores[valid], classes[valid])
    order = np.argsort(-scores, kind='stable')[:3000]
    if quality_filter:
        if head_geometry:
            head = np.array([names[int(c)].casefold() == 'head' for c in classes[order]], dtype=bool)
            sizes = boxes[order, 2:] - boxes[order, :2]
            order = order[~head | ((sizes[:, 0] >= .45 * sizes[:, 1]) & (sizes[:, 0] <= 2 * sizes[:, 1]))]
        candidates = set()
        for class_id in np.unique(classes[order]):
            indices = order[classes[order] == class_id]
            rects = boxes[indices].copy()
            rects[:, 2:] -= rects[:, :2]
            selected = cv2.dnn.NMSBoxes(rects.tolist(), scores[indices].tolist(), 0.0, .95)
            candidates.update(indices[np.asarray(selected, dtype=int).ravel()].tolist())
        order = np.asarray([index for index in order if index in candidates][:512], dtype=int)
    if head_geometry:
        is_head = np.array([names[int(c)].casefold() == 'head' for c in classes[order]], dtype=bool)
        bodies = boxes[order[[names[int(c)].casefold() == 'body' for c in classes[order]]]]
        heads = boxes[order[is_head]]
        sizes = heads[:, 2:] - heads[:, :2]
        plausible = (sizes[:, 0] >= 0.45 * sizes[:, 1]) & (sizes[:, 0] <= 2 * sizes[:, 1])
        if len(bodies) and len(heads):
            cx, cy = ((heads[:, :2] + heads[:, 2:]) / 2).T
            bw, bh = (bodies[:, 2:] - bodies[:, :2]).T
            nearby = (cx[:, None] >= bodies[:, 0]) & (cx[:, None] <= bodies[:, 2]) & (cy[:, None] >= bodies[:, 1] - 0.15 * bh) & (cy[:, None] <= bodies[:, 3])
            good = (cy[:, None] <= bodies[:, 1] + 0.4 * bh) & (sizes[:, 1, None] <= 0.4 * bh) | (sizes[:, 1, None] <= 0.18 * bh) & (sizes[:, 0, None] <= 0.5 * bw)
            plausible &= ~nearby.any(axis=1) | (nearby & good).any(axis=1)
        accepted = np.ones(len(order), dtype=bool)
        accepted[is_head] = plausible
        order = order[accepted]
    if quality_filter:
        from detection_quality import select_detections
        selected = select_detections(boxes[order], scores[order], [names[int(classes[index])] for index in order], iou=min(iou, .65), head_iou=.3 if head_iou is None else head_iou)
        keep = order[selected].tolist()
    elif end_to_end:
        keep = order[:300]
    else:
        selected = set()
        for class_id in np.unique(classes[order]):
            indices = order[classes[order] == class_id]
            rects = boxes[indices].copy()
            rects[:, 2:] -= rects[:, :2]
            threshold = head_iou if head_iou is not None and names[int(class_id)].casefold() == 'head' else iou
            kept = cv2.dnn.NMSBoxes(rects.tolist(), scores[indices].tolist(), 0.0, threshold)
            selected.update(indices[np.asarray(kept, dtype=int).ravel()].tolist())
        keep = [index for index in order if index in selected][:300]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - geometry.left) / geometry.width
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - geometry.top) / geometry.height
    np.clip(boxes, 0, 1, out=boxes)
    return [(*map(float, boxes[index]), names[int(classes[index])], f'{names[int(classes[index])]} {float(scores[index]):.0%}', *((float(scores[index]),) if return_scores else ())) for index in keep if boxes[index, 2] > boxes[index, 0] and boxes[index, 3] > boxes[index, 1]]

class DirectMLDetector:

    def __init__(self, model_path: Path, device_id=0, *, profile_prefix=None, runtime='native'):
        if device_id < 0:
            raise ValueError('DirectML device ID must be nonnegative')
        if 'DmlExecutionProvider' not in ort.get_available_providers():
            raise RuntimeError('DirectML is unavailable. Run setup_directml.bat; CPU inference is disabled.')
        if not Path(model_path).is_file():
            raise FileNotFoundError(f'Missing ONNX model: {model_path}. Run build_onnx.bat first.')
        cv2.setNumThreads(1)
        self.runtime = runtime
        if runtime == 'native':
            from native_dml import NativeDMLSession
            self.session = NativeDMLSession(model_path, device_id, profile_prefix)
        elif runtime == 'python':
            options = ort.SessionOptions()
            options.enable_mem_pattern = False
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.add_session_config_entry('session.intra_op.allow_spinning', '0')
            options.add_session_config_entry('session.inter_op.allow_spinning', '0')
            options.add_session_config_entry('session.disable_cpu_ep_fallback', '1')
            if profile_prefix:
                options.enable_profiling = True
                options.profile_file_prefix = str(profile_prefix)
            self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=[('DmlExecutionProvider', {'device_id': str(device_id)})])
        else:
            raise ValueError(f'Unsupported DirectML runtime: {runtime}')
        self.session.disable_fallback()
        if self.session.get_providers()[0] != 'DmlExecutionProvider':
            raise RuntimeError('DirectML session creation failed. CPU inference is disabled.')
        inputs, outputs = (self.session.get_inputs(), self.session.get_outputs())
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError('Expected a single-input, single-output YOLO detection model')
        shape = inputs[0].shape
        if len(shape) != 4 or shape[:2] != [1, 3] or (not isinstance(shape[2], int)) or (not isinstance(shape[3], int)) or (min(shape[2:]) < 32) or (inputs[0].type != 'tensor(float)'):
            raise ValueError(f'Expected static float32 [1, 3, height, width] input; got {shape}')
        self.size, self.input_name = (max(shape[2:]), inputs[0].name)
        self.input_size = (shape[3], shape[2])
        metadata = self.session.get_modelmeta().custom_metadata_map
        self.head_confidence = float(metadata.get('head_confidence', '.25'))
        if not 0 < self.head_confidence <= 1:
            raise ValueError('Model head_confidence must be between 0 and 1')
        if metadata.get('task') != 'detect':
            raise ValueError('Only YOLO object detection models are supported')
        self.names = {int(k): str(v) for k, v in ast.literal_eval(metadata.get('names', '{}')).items()}
        if not self.names or set(self.names) != set(range(len(self.names))):
            raise ValueError('The ONNX model must contain contiguous class names metadata')
        output_shape = outputs[0].shape
        if len(output_shape) != 3 or output_shape[0] != 1:
            raise ValueError(f'Unsupported detection output: {output_shape}')
        self.end_to_end = output_shape[1] != 4 + len(self.names) and output_shape[2] == 6
        if not self.end_to_end and output_shape[1] != 4 + len(self.names):
            raise ValueError(f'Unsupported detection output: {output_shape}')

    def predict(self, frame, view='full'):
        selected, region = detection_view(frame, view)
        tensor, geometry = prepare_frame(selected, self.input_size)
        output = self.session.run(None, {self.input_name: tensor})[0]
        detections = decode_output(output, geometry, self.names, end_to_end=self.end_to_end, head_iou=0.3, head_geometry=True, head_confidence=self.head_confidence, quality_filter=True)
        return remap_detections(detections, region, frame.shape)
