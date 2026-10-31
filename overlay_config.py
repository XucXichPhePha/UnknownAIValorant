from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class OverlayConfig:
    anti_recording: bool = True
    model_path: Path | None = None
    max_view_fps: int = 60
    max_render_fps: int = 60
    show_debug_info: bool = False
    capture_view: str | None = None
    motion_assistance: bool = True


class ConfigLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError('Config keys must be unique names')
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def load_config(path, *, required=False):
    path = Path(path).expanduser().resolve()
    try:
        if path.stat().st_size > 65536:
            raise ValueError('Overlay config must be smaller than 64 KB')
        data = yaml.load(path.read_text(encoding='utf-8-sig'), Loader=ConfigLoader)
    except FileNotFoundError:
        if required:
            raise ValueError(f'Overlay config not found: {path}') from None
        return OverlayConfig()
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f'Could not read overlay config {path}: {exc}') from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError('Overlay config must contain named settings')
    unknown = set(data) - set(OverlayConfig.__dataclass_fields__)
    if unknown:
        raise ValueError('Unknown overlay setting: ' + ', '.join(sorted(unknown)))
    for name in ('anti_recording', 'show_debug_info', 'motion_assistance'):
        if name in data and type(data[name]) is not bool:
            raise ValueError(f'{name} must be true or false')
    for name in ('max_view_fps', 'max_render_fps'):
        if name in data and (type(data[name]) is not int or not 1 <= data[name] <= 240):
            raise ValueError(f'{name} must be a whole number from 1 to 240')
    if data.get('capture_view') is not None and data['capture_view'] not in ('fast', 'compact', 'tight', 'focus', 'full'):
        raise ValueError('capture_view must be fast, compact, tight, focus, full, or null')
    model = data.get('model_path')
    if model is not None and not isinstance(model, str):
        raise ValueError('model_path must be an ONNX/PT file path or null')
    if model and model.strip():
        model = Path(model.strip()).expanduser()
        if model.suffix.lower() not in ('.onnx', '.pt'):
            raise ValueError('model_path must select an .onnx or .pt file')
        data['model_path'] = (path.parent / model).resolve() if not model.is_absolute() else model.resolve()
    elif 'model_path' in data:
        data['model_path'] = None
    return OverlayConfig(**data)
