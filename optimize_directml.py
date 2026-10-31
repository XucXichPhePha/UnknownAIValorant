import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile


SHAPES = {'fast': (320, 224), 'compact': (224, 160)}


def export_optimized(checkpoint, destination, view='fast'):
    if view not in SHAPES:
        raise ValueError('Optimized view must be fast or compact')
    checkpoint, destination = Path(checkpoint).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f'Model already exists: {destination}')
    if not checkpoint.is_file() or checkpoint.suffix.casefold() != '.pt':
        raise ValueError('A matching original PT checkpoint is required')
    if not destination.parent.is_dir() or destination.suffix.casefold() != '.onnx':
        raise ValueError('Choose an ONNX output in an existing directory')
    import onnx
    import torch
    from onnxruntime.transformers.float16 import convert_float_to_float16
    from onnxruntime.transformers.onnx_model import OnnxModel
    width, height = SHAPES[view]
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='task-dml-export-', dir=destination.parent) as temporary:
        environment = {'YOLO_CONFIG_DIR': temporary, 'YOLO_OFFLINE': 'true', 'YOLO_AUTOINSTALL': 'false'}
        previous = {key: os.environ.get(key) for key in environment}
        os.environ.update(environment)
        try:
            from ultralytics import YOLO
            source = Path(temporary)/'source.pt'
            shutil.copyfile(checkpoint, source)
            network = YOLO(str(source))
            if network.task != 'detect' or network.names != {0: 'Head', 1: 'Body'}:
                raise ValueError('The model must detect exactly Head and Body')
            exported = network.export(format='onnx', imgsz=(height, width), opset=17,
                                      dynamic=False, simplify=False, device='cpu', batch=1)
            model = convert_float_to_float16(onnx.load(exported), keep_io_types=True)
            OnnxModel(model).topological_sort()
            onnx.checker.check_model(model)
            if (len(model.graph.input) != 1 or len(model.graph.output) != 1
                    or model.graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
                    or model.graph.output[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT
                    or [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim] != [1, 3, height, width]):
                raise ValueError('Optimized model must preserve static float32 input/output boundaries')
            metadata = {entry.key: entry.value for entry in model.metadata_props}
            metadata.update(overlay_capture_view=view, optimized_precision='mixed-fp16',
                            source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            onnx.helper.set_model_props(model, metadata)
            converted = Path(temporary)/'converted.onnx'
            onnx.save(model, str(converted))
            created = False
            try:
                with destination.open('xb') as handle:
                    created = True
                    with converted.open('rb') as stream:
                        shutil.copyfileobj(stream, handle)
            except BaseException:
                if created:
                    destination.unlink(missing_ok=True)
                raise
        except ModuleNotFoundError as exc:
            raise RuntimeError(f'Export requires installed training dependencies; automatic installation is disabled. Missing: {exc.name}') from exc
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return destination


def main():
    parser = argparse.ArgumentParser(description='Export a rectangular FP16 DirectML model with float32 inputs and outputs')
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--view', choices=tuple(SHAPES), default='fast')
    args = parser.parse_args()
    print(export_optimized(args.checkpoint, args.output, args.view))


if __name__ == '__main__':
    main()
