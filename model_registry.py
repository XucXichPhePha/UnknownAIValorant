from __future__ import annotations
import contextlib
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def replace_file(source, destination):
    for attempt in range(8):
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if getattr(error, 'winerror', None) not in (5, 32, 33) or attempt == 7:
                raise
            time.sleep(min(0.025 * 2 ** attempt, 0.4))

def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f'write-{secrets.token_hex(8)}.tmp')
    try:
        with temporary.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default

def process_alive(pid):
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(4096, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except ProcessLookupError:
        return False

@contextlib.contextmanager
def file_lock(path, timeout=10):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        deadline = time.monotonic() + timeout
        while True:
            try:
                stream.seek(0)
                if os.name == 'nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (OSError, BlockingIOError):
                if time.monotonic() >= deadline:
                    raise ValueError('Another local operation is busy; try again')
                time.sleep(0.03)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_UN)

def _inspect_onnx(path):
    import ast
    import onnx
    model = onnx.load(str(path), load_external_data=False)
    if any((t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer)):
        raise ValueError('Models must contain their tensor data in one ONNX file')
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError('Expected one model input and one detection output')
    tensor = model.graph.input[0].type.tensor_type
    shape = [int(d.dim_value) for d in tensor.shape.dim]
    output = [int(d.dim_value) for d in model.graph.output[0].type.tensor_type.shape.dim]
    if tensor.elem_type != onnx.TensorProto.FLOAT or len(shape) != 4 or shape[:2] != [1, 3] or any((d < 32 or d > 1280 or d % 32 for d in shape[2:])):
        raise ValueError('Expected a static float32 [1, 3, height, width] input in multiples of 32')
    if model.graph.output[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT or len(output) != 3 or output[:2] != [1, 6] or (output[2] <= 0):
        raise ValueError('Expected a static Head/Body [1, 6, anchors] output')
    metadata = {item.key: item.value for item in model.metadata_props}
    if metadata.get('task') != 'detect':
        raise ValueError('Only YOLO object detection models are supported')
    try:
        confidence = float(metadata.get('head_confidence', '.25'))
    except (TypeError, ValueError) as exc:
        raise ValueError('Model head_confidence must be a finite number greater than 0 and at most 1') from exc
    if not math.isfinite(confidence) or not 0 < confidence <= 1:
        raise ValueError('Model head_confidence must be a finite number greater than 0 and at most 1')
    try:
        names = ast.literal_eval(metadata.get('names', '{}'))
    except (ValueError, SyntaxError) as exc:
        raise ValueError('Invalid Head/Body model class metadata') from exc
    if names != {0: 'Head', 1: 'Body'}:
        raise ValueError('Model classes must be exactly 0: Head and 1: Body')
    return {'input_shape': shape}

def validate_onnx(path):
    try:
        return _inspect_onnx(path)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f'ONNX validation failed: {exc}') from exc

class ModelRegistry:
    BUILTINS = (('builtin-focus', 'Original · focus', 'best.onnx', 'focus', [1, 3, 320, 320]), ('builtin-tight', 'Original · tight', 'best-focus-256x192.onnx', 'tight', [1, 3, 192, 256]), ('builtin-fast', 'Original · fast', 'best-focus-192x128.onnx', 'fast', [1, 3, 128, 192]))

    def __init__(self, project, *, validator=None):
        self.project = Path(project).resolve()
        self.root = self.project / 'model-library'
        self.validator = validator or validate_onnx

    def _prepare(self):
        if self.root.is_symlink() or self.root.resolve().parent != self.project:
            raise ValueError('model-library must stay inside the project')
        self.root.mkdir(exist_ok=True)

    def _entries(self):
        entries = []
        for model_id, label, filename, view, shape in self.BUILTINS:
            path = self.project / filename
            if path.is_file() and (not path.is_symlink()):
                entries.append(dict(id=model_id, label=label, path=str(path), view=view, family='builtin', input_shape=shape, created_at=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()))
        manifest = read_json(self.root / 'models.json', {'models': []})
        for entry in manifest.get('models', []) if isinstance(manifest, dict) else []:
            if not isinstance(entry, dict) or not re.fullmatch('model-[a-f0-9]{24}', str(entry.get('id', ''))):
                continue
            if not all((key in entry for key in ('label', 'view', 'input_shape', 'created_at'))) or entry['view'] not in ('full', 'focus', 'tight', 'fast'):
                continue
            path = self.root / entry['id'] / 'model.onnx'
            if path.resolve().parent != (self.root / entry['id']).absolute() or path.is_symlink():
                continue
            if path.is_file():
                entries.append({**entry, 'path': str(path)})
        return entries

    def _active_id(self, entries):
        selected = read_json(self.root / 'active.json', {})
        selected_id = selected.get('model_id') if isinstance(selected, dict) else None
        ids = {entry['id'] for entry in entries}
        if selected_id in ids:
            return selected_id
        return next((model_id for model_id in ('builtin-fast', 'builtin-tight', 'builtin-focus') if model_id in ids), None)

    def list_models(self):
        entries = self._entries()
        active_id = self._active_id(entries)
        keys = ('id', 'label', 'path', 'view', 'input_shape', 'created_at')
        return [{**{key: entry[key] for key in keys}, 'is_active': entry['id'] == active_id,
                 'can_delete': not entry['id'].startswith('builtin-') and entry['id'] != active_id,
                 'delete_reason': 'Included models cannot be deleted' if entry['id'].startswith('builtin-') else ('Switch the active default model before deleting it' if entry['id'] == active_id else None),
                 'formats': self._formats(entry)} for entry in entries]

    def delete(self, model_id):
        from local_trash import LocalTrash
        self._prepare()
        with file_lock(self.root / 'registry.lock'):
            entry = self.resolve(model_id)
            if not entry['can_delete']:
                raise ValueError(entry['delete_reason'])
            manifest = read_json(self.root / 'models.json')
            if not isinstance(manifest, dict) or not isinstance(manifest.get('models'), list):
                raise ValueError('Model library metadata is damaged; existing models were preserved')
            original = next((item for item in manifest['models'] if item.get('id') == model_id), None)
            if original is None:
                raise ValueError('Unknown managed model')
            trash = LocalTrash(self.project)
            record = trash.put('model', self.root / model_id, entry['label'], model_id=model_id, model_entry=original)
            try:
                atomic_json(self.root / 'models.json', {**manifest, 'models': [item for item in manifest['models'] if item.get('id') != model_id]})
            except BaseException:
                trash.restore(record['id'], kind='model')
                raise
            return dict(deleted=True, trash_id=record['id'], model_id=model_id)

    def restore(self, trash_id):
        from local_trash import LocalTrash
        self._prepare()
        with file_lock(self.root / 'registry.lock'):
            def commit(record, path):
                manifest = read_json(self.root / 'models.json', {'models': []})
                if not isinstance(manifest, dict) or not isinstance(manifest.get('models'), list):
                    raise ValueError('Model library metadata is damaged; the deleted item was preserved')
                original = record.get('model_entry')
                if not isinstance(original, dict) or original.get('id') != record['model_id']:
                    raise ValueError('The deleted model metadata is invalid')
                if any(item.get('id') == original['id'] and item != original for item in manifest['models']):
                    raise ValueError('This model ID already exists with different metadata')
                restored = {**manifest, 'models': [item for item in manifest['models'] if item.get('id') != original['id']] + [original]}
                entry = {**original, 'path': str(path / 'model.onnx')}
                self.validator(Path(entry['path']))
                if _sha256(entry['path']) != original.get('sha256'):
                    raise ValueError('The deleted model content changed')
                atomic_json(self.root / 'models.json', restored)
                return entry
            result = LocalTrash(self.project).restore(trash_id, kind='model', commit=commit)
            return self.resolve(result['id'])

    def _artifact_path(self, entry, format):
        if format not in ('onnx', 'pt'):
            raise ValueError('Choose onnx or pt format')
        builtin = entry['id'].startswith('builtin-')
        parent = self.project if builtin else self.root / entry['id']
        if format == 'onnx':
            path = Path(entry['path'])
        else:
            if not builtin and not entry.get('checkpoint'):
                raise FileNotFoundError('This model has no matching PT checkpoint')
            path = parent / ('best.pt' if builtin else 'checkpoint.pt')
        if path.is_symlink() or path.resolve().parent != parent.absolute() or not path.is_file():
            raise FileNotFoundError('The requested model file is unavailable')
        return path

    def _formats(self, entry):
        formats = []
        for format in ('onnx', 'pt'):
            try:
                self._artifact_path(entry, format)
                formats.append(format)
            except OSError:
                pass
        return formats

    def artifact_path(self, model_id, format):
        if not isinstance(model_id, str):
            raise ValueError('Choose a known model ID')
        entry = next((entry for entry in self._entries() if entry['id'] == model_id), None)
        if entry is None:
            raise ValueError('Unknown model ID; refresh the model library')
        return self._artifact_path(entry, format)

    def active(self):
        return next((entry for entry in self.list_models() if entry['is_active']), None)

    def resolve(self, model_id):
        if not isinstance(model_id, str):
            raise ValueError('Choose a known model ID')
        for entry in self.list_models():
            if entry['id'] == model_id:
                return entry
        raise ValueError('Unknown model ID; refresh the model library')

    def select(self, model_id):
        self._prepare()
        with file_lock(self.root / 'registry.lock'):
            entry = self.validate(model_id)
            atomic_json(self.root / 'active.json', {'model_id': model_id, 'selected_at': utc_now()})
        return {**entry, 'is_active': True}

    def validate(self, model_id):
        entry = self.resolve(model_id)
        path = Path(entry['path'])
        checked = self.validator(path)
        if list(checked['input_shape']) != entry['input_shape']:
            raise ValueError('Model shape changed; the previous default was preserved')
        if model_id.startswith('model-') and (not hashlib.sha256(path.read_bytes()).hexdigest().startswith(model_id[6:])):
            raise ValueError('Managed model content changed; the previous default was preserved')
        return entry

    def for_view(self, model_id, view):
        if view not in ('full', 'focus', 'tight', 'fast'):
            raise ValueError('Unsupported model view')
        entries = self._entries()
        current = next((entry for entry in entries if entry['id'] == model_id), None)
        if current is None:
            raise ValueError('Unknown model ID; refresh the model library')
        if current['view'] == view:
            return self.resolve(current['id'])
        family = current.get('family', current['id'])
        siblings = [entry for entry in entries if entry.get('family', entry['id']) == family]
        candidates = [entry for entry in siblings if entry['view'] == view]
        if not candidates and view == 'full' and current['view'] == 'focus':
            return self.resolve(current['id'])
        if not candidates and view in ('full', 'focus'):
            candidates = [entry for entry in siblings if entry['view'] == ('focus' if view == 'full' else 'full')]
        sibling = max(candidates, key=lambda entry: entry['input_shape'][2] * entry['input_shape'][3], default=None)
        return self.resolve(sibling['id']) if sibling else None

    def register(self, path, label=None, view=None, checkpoint=None):
        source = Path(path).resolve()
        if not source.is_file() or source.suffix.lower() != '.onnx':
            raise ValueError('A local ONNX model file is required')
        validation = self.validator(source)
        shape = list(validation['input_shape'])
        recommended = {(128, 192): 'fast', (192, 256): 'tight'}.get(tuple(shape[2:]), 'focus')
        view = recommended if view is None else view
        if view not in ('full', 'focus', 'tight', 'fast'):
            raise ValueError('Unsupported model view')
        if view in ('tight', 'fast') and view != recommended:
            raise ValueError('Model input shape does not match the requested view')
        if label is not None and (not isinstance(label, str) or not label.strip() or len(label) > 160):
            raise ValueError('Model label must contain 1–160 characters')
        checkpoint = Path(checkpoint).resolve() if checkpoint is not None else None
        if checkpoint is not None and (checkpoint.suffix.lower() != '.pt' or not checkpoint.is_file()):
            raise ValueError('A local checkpoint file is required')
        checkpoint_digest = _sha256(checkpoint) if checkpoint is not None else None
        digest = _sha256(source)
        model_id = f'model-{digest[:24]}'
        self._prepare()
        with file_lock(self.root / 'registry.lock'):
            manifest = read_json(self.root / 'models.json', {'models': []})
            if not isinstance(manifest, dict) or not isinstance(manifest.get('models'), list):
                raise ValueError('Model library metadata is damaged; existing models were preserved')
            existing = next((entry for entry in manifest['models'] if entry.get('id') == model_id), None)
            if existing is not None:
                current = self.validate(model_id)
                if _sha256(current['path']) != digest:
                    raise ValueError('The managed ONNX differs from the imported model; existing files were preserved')
                if checkpoint is not None:
                    destination = self.root / model_id
                    target = destination / 'checkpoint.pt'
                    family = 'checkpoint-' + checkpoint_digest
                    if destination.is_symlink() or destination.resolve() != destination.absolute() or target.is_symlink() or target.resolve().parent != destination:
                        raise ValueError('The matching checkpoint must stay inside its managed model folder')
                    if existing.get('checkpoint') and existing.get('family') != family:
                        raise ValueError('This ONNX already has a different matching checkpoint; existing files were preserved')
                    created = False
                    try:
                        if target.exists():
                            if not target.is_file() or _sha256(target) != checkpoint_digest:
                                raise ValueError('The existing matching checkpoint differs; existing files were preserved')
                        else:
                            with target.open('xb') as outgoing, checkpoint.open('rb') as incoming:
                                created = True
                                shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
                                outgoing.flush()
                                os.fsync(outgoing.fileno())
                            if _sha256(target) != checkpoint_digest:
                                raise ValueError('The checkpoint changed during registration')
                        if not existing.get('checkpoint'):
                            existing.update(checkpoint=True, family=family)
                            atomic_json(self.root / 'models.json', manifest)
                    except BaseException:
                        if created:
                            target.unlink(missing_ok=True)
                        raise
                return self.resolve(model_id)
            destination = self.root / model_id
            if destination.resolve().parent != self.root:
                raise ValueError('Model destination leaves the library')
            destination.mkdir(exist_ok=False)
            try:
                model_path = destination / 'model.onnx'
                shutil.copyfile(source, model_path)
                if _sha256(model_path) != digest:
                    raise ValueError('Model changed during registration')
                self.validator(model_path)
                entry = dict(id=model_id, label=label or source.stem, view=view, input_shape=shape, created_at=utc_now(), sha256=digest)
                if checkpoint is not None:
                    shutil.copyfile(checkpoint, destination / 'checkpoint.pt')
                    if _sha256(destination / 'checkpoint.pt') != checkpoint_digest:
                        raise ValueError('The checkpoint changed during registration')
                    entry['checkpoint'] = True
                    entry['family'] = 'checkpoint-' + checkpoint_digest
                manifest['models'].append(entry)
                atomic_json(self.root / 'models.json', manifest)
            except BaseException:
                shutil.rmtree(destination)
                raise
        return self.resolve(model_id)

    def checkpoints(self):
        values = []
        if (self.project / 'best.pt').is_file():
            values.append({'id': 'builtin-best', 'label': 'Original local best.pt'})
        for entry in self._entries():
            if entry.get('checkpoint') and (self.root / entry['id'] / 'checkpoint.pt').is_file():
                values.append({'id': entry['id'], 'label': entry['label']})
        return values

    def checkpoint_path(self, checkpoint_id):
        if not isinstance(checkpoint_id, str) or checkpoint_id not in {entry['id'] for entry in self.checkpoints()}:
            raise ValueError('Choose a known local checkpoint')
        return (self.project / 'best.pt' if checkpoint_id == 'builtin-best' else self.root / checkpoint_id / 'checkpoint.pt').resolve()
