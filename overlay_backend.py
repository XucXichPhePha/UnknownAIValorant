from pathlib import Path
import numpy as np


class UltralyticsDetector:
    def __init__(self, model_path, device_id=0, *, model_factory=None):
        import torch
        if model_factory is None:
            from ultralytics import YOLO
            model_factory = YOLO
        torch.set_num_threads(2)
        self.model = model_factory(str(model_path))
        if self.model.task != 'detect' or self.model.names != {0: 'Head', 1: 'Body'}:
            raise ValueError('PT overlay models must use classes exactly 0=Head and 1=Body')
        self.names = self.model.names
        self.device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
        self.backend_label = 'Ultralytics / ' + ('CUDA GPU ' + str(device_id) if self.device != 'cpu' else 'CPU')
        size = getattr(self.model, 'overrides', {}).get('imgsz') or getattr(self.model.model, 'args', {}).get('imgsz', 320)
        size = [size, size] if type(size) is int else size
        if not isinstance(size, (list, tuple)) or len(size) != 2 or any(type(value) is not int or not 32 <= value <= 1280 or value % 32 for value in size):
            raise ValueError('PT inference dimensions must be multiples of 32 from 32 to 1280')
        self.input_size = (size[1], size[0])
        self.size = max(size)

    def predict(self, frame, view='full'):
        from directml_backend import detection_view, filter_head_detections, remap_detections
        from detection_quality import select_detections
        selected, region = detection_view(frame, view)
        result = self.model.predict(source=np.ascontiguousarray(selected[:, :, :3]), device=self.device,
                                    imgsz=[self.input_size[1], self.input_size[0]], conf=.25, iou=.65,
                                    max_det=300, rect=False, verbose=False, save=False)[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        if boxes.ndim != 2 or boxes.shape[1] != 4 or classes.shape != (len(boxes),) or scores.shape != (len(boxes),) or not all(np.isfinite(value).all() for value in (boxes, classes, scores)):
            raise ValueError('PT model returned invalid detection coordinates')
        if not np.isin(classes, (0, 1)).all() or not ((scores >= 0) & (scores <= 1)).all():
            raise ValueError('PT model returned invalid classes or confidence')
        labels = [self.names[int(value)] for value in classes]
        keep = select_detections(boxes, scores, labels)
        height, width = selected.shape[:2]
        normalized = np.clip(boxes / [width, height, width, height], 0, 1)
        detections = [(*map(float, normalized[index]), labels[index], f'{labels[index]} {scores[index]:.0%}') for index in keep
                      if scores[index] >= .25 and normalized[index, 2] > normalized[index, 0] and normalized[index, 3] > normalized[index, 1]]
        detections = filter_head_detections(detections, selected.shape)
        return remap_detections(detections, region, frame.shape)


def create_detector(model_path, device_id=0, *, runtime='native'):
    suffix = Path(model_path).suffix.lower()
    if suffix == '.pt':
        return UltralyticsDetector(model_path, device_id)
    if suffix == '.onnx':
        from directml_backend import DirectMLDetector
        model = DirectMLDetector(model_path, device_id, runtime=runtime)
        model.backend_label = f'DirectML / GPU {device_id}'
        return model
    raise ValueError('Overlay models must use .onnx or .pt format')


def model_banner(model_path, model):
    width, height = model.input_size
    return f'Model: {Path(model_path).resolve()} | input: {width}x{height} | {model.backend_label}'
