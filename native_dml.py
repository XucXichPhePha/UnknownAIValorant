"""Blocking DirectML sessions through the installed ONNX Runtime 1.24.4 C API.

The Python DML provider supplies its own spinning execution context. Creating
the native provider directly lets ep.dml.enable_cpu_sync_spinning=0 take effect.
No other provider or Python inference path is used by this module.
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
import ctypes as C
import operator
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np


_ORT_VERSION = '1.24.4'
_API_VERSION = 24
_P, _S, _I = C.c_void_p, C.c_size_t, C.c_int
_PP = C.POINTER(_P)
_I64P = C.POINTER(C.c_int64)
_NAMES = C.POINTER(C.c_char_p)

# Stable, append-only OrtApi table entries from Microsoft's MIT-licensed headers:
# https://github.com/microsoft/onnxruntime/blob/v1.24.4/include/onnxruntime/core/session/onnxruntime_c_api.h
# https://github.com/microsoft/onnxruntime/blob/v1.24.4/include/onnxruntime/core/providers/dml/dml_provider_factory.h
# Values are pointer slots, not byte offsets. Only functions used below are bound;
# no downloaded headers, exported Python internals, or external DLLs are needed.
_FUNCTIONS = {
    'GetErrorMessage': (2, C.c_char_p, _P),
    'CreateEnv': (3, _P, _I, C.c_char_p, _PP),
    'CreateSession': (7, _P, _P, C.c_wchar_p, _P, _PP),
    'Run': (9, _P, _P, _P, _NAMES, _PP, _S, _NAMES, _S, _PP),
    'CreateSessionOptions': (10, _P, _PP),
    'SetSessionExecutionMode': (13, _P, _P, _I),
    'EnableProfiling': (14, _P, _P, C.c_wchar_p),
    'DisableMemPattern': (17, _P, _P),
    'SetIntraOpNumThreads': (24, _P, _P, _I),
    'SetInterOpNumThreads': (25, _P, _P, _I),
    'SessionGetInputCount': (30, _P, _P, C.POINTER(_S)),
    'SessionGetOutputCount': (31, _P, _P, C.POINTER(_S)),
    'SessionGetInputTypeInfo': (33, _P, _P, _S, _PP),
    'SessionGetOutputTypeInfo': (34, _P, _P, _S, _PP),
    'SessionGetInputName': (36, _P, _P, _S, _P, _PP),
    'SessionGetOutputName': (37, _P, _P, _S, _P, _PP),
    'CreateTensorWithDataAsOrtValue': (49, _P, _P, _P, _S, _I64P, _S, _I, _PP),
    'CastTypeInfoToTensorInfo': (55, _P, _P, _PP),
    'GetTensorElementType': (60, _P, _P, C.POINTER(_I)),
    'GetDimensionsCount': (61, _P, _P, C.POINTER(_S)),
    'GetDimensions': (62, _P, _P, _I64P, _S),
    'CreateCpuMemoryInfo': (69, _P, _I, _I, _PP),
    'AllocatorFree': (76, _P, _P, _P),
    'GetAllocatorWithDefaultOptions': (78, _P, _PP),
    'ReleaseEnv': (92, None, _P),
    'ReleaseStatus': (93, None, _P),
    'ReleaseMemoryInfo': (94, None, _P),
    'ReleaseSession': (95, None, _P),
    'ReleaseValue': (96, None, _P),
    'ReleaseTypeInfo': (98, None, _P),
    'ReleaseSessionOptions': (100, None, _P),
    'SessionEndProfiling': (110, _P, _P, _P, _PP),
    'SessionGetModelMetadata': (111, _P, _P, _PP),
    'ModelMetadataLookupCustomMetadataMap': (116, _P, _P, _P, C.c_char_p, _PP),
    'ReleaseModelMetadata': (118, None, _P),
    'ModelMetadataGetCustomMetadataMapKeys': (123, _P, _P, _P, C.POINTER(_PP), C.POINTER(C.c_int64)),
    'AddSessionConfigEntry': (130, _P, _P, C.c_char_p, C.c_char_p),
    'GetExecutionProviderApi': (195, _P, C.c_char_p, C.c_uint32, _PP),
}


class _OrtApi:
    def __init__(self):
        if sys.platform != 'win32':
            raise RuntimeError('Native DirectML inference requires Windows')
        import onnxruntime as ort
        if ort.__version__ != _ORT_VERSION:
            raise RuntimeError(f'Native DirectML requires onnxruntime-directml=={_ORT_VERSION}; got {ort.__version__}')
        library = Path(ort.__file__).resolve().parent / 'capi' / 'onnxruntime.dll'
        if not library.is_file():
            raise RuntimeError(f'The installed DirectML runtime DLL is missing: {library}')
        self._dll = C.WinDLL(str(library), winmode=0x1100)
        export = self._dll.OrtGetApiBase
        export.argtypes, export.restype = [], C.POINTER(_P)
        base = export()
        if not base or not base[0] or not base[1]:
            raise RuntimeError('ONNX Runtime returned an invalid C API base')
        prototype = C.WINFUNCTYPE
        version = prototype(C.c_char_p)(base[1])()
        if version != _ORT_VERSION.encode():
            raise RuntimeError(f'ONNX Runtime DLL version must be {_ORT_VERSION}; got {version!r}')
        api = prototype(_P, C.c_uint32)(base[0])(_API_VERSION)
        if not api:
            raise RuntimeError(f'ONNX Runtime C API {_API_VERSION} is unavailable')
        table = C.cast(api, C.POINTER(_P))
        for name, (index, result, *arguments) in _FUNCTIONS.items():
            if not table[index]:
                raise RuntimeError(f'ONNX Runtime C API is missing {name}')
            setattr(self, name, prototype(result, *arguments)(table[index]))
        provider = _P()
        self.check(self.GetExecutionProviderApi(b'DML', _API_VERSION, C.byref(provider)))
        if not provider:
            raise RuntimeError('DirectML provider API is unavailable; CPU inference is disabled')
        append = C.cast(provider, C.POINTER(_P))[0]
        if not append:
            raise RuntimeError('DirectML provider creation API is unavailable')
        self.append_directml = prototype(_P, _P, _I)(append)

    def check(self, status):
        if status:
            try:
                message = self.GetErrorMessage(status).decode('utf-8', 'replace')
            finally:
                self.ReleaseStatus(status)
            raise RuntimeError(f'ONNX Runtime DirectML: {message}')


def _load_api():
    return _OrtApi()


class NativeDMLSession:
    """Single-input/output static FP32 session with reusable CPU OrtValues.

    Run synchronously copies each supplied feed into the same CPU input buffer.
    ORT transfers fresh input on every native Run; the returned NumPy arrays are
    independent copies, safe after another run or after closing this session.
    """
    def __init__(self, model_path, device_id=0, profile_prefix=None):
        self._lock = threading.Lock()
        self._resources = ExitStack()
        self._closed = False
        self._input_buffer = self._output_buffer = None
        self._profile_path = None
        self._profiling = profile_prefix is not None
        try:
            device_id = operator.index(device_id)
            if not 0 <= device_id <= 2**31 - 1:
                raise ValueError('DirectML device ID must be a nonnegative 32-bit integer')
            model_path = Path(model_path).expanduser().resolve()
            if not model_path.is_file():
                raise FileNotFoundError(f'Missing ONNX model: {model_path}')
            self._api = _load_api()
            env = self._handle('CreateEnv', 'ReleaseEnv', 2, b'native-dml')
            options = self._handle('CreateSessionOptions', 'ReleaseSessionOptions')
            for name, value in (('SetSessionExecutionMode', 0), ('SetIntraOpNumThreads', 1),
                                ('SetInterOpNumThreads', 1)):
                self._api.check(getattr(self._api, name)(options, value))
            self._api.check(self._api.DisableMemPattern(options))
            for key, value in ((b'session.disable_cpu_ep_fallback', b'1'),
                               (b'session.intra_op.allow_spinning', b'0'),
                               (b'session.inter_op.allow_spinning', b'0'),
                               (b'ep.dml.enable_cpu_sync_spinning', b'0')):
                self._api.check(self._api.AddSessionConfigEntry(options, key, value))
            if self._profiling:
                self._api.check(self._api.EnableProfiling(options, str(profile_prefix)))
            self._api.check(self._api.append_directml(options, device_id))
            self._session = self._handle('CreateSession', 'ReleaseSession', env, str(model_path), options)
            self._allocator = _P()
            self._api.check(self._api.GetAllocatorWithDefaultOptions(C.byref(self._allocator)))
            if (self._scalar('SessionGetInputCount', _S, self._session) != 1
                    or self._scalar('SessionGetOutputCount', _S, self._session) != 1):
                raise ValueError('Expected a single-input, single-output model')
            self._input_name, self._input_shape = self._read_node('Input')
            self._output_name, self._output_shape = self._read_node('Output')
            self._metadata = self._read_metadata()
            memory = self._handle('CreateCpuMemoryInfo', 'ReleaseMemoryInfo', 1, 0)
            self._input_buffer = np.empty(self._input_shape, dtype=np.float32)
            self._output_buffer = np.empty(self._output_shape, dtype=np.float32)
            input_value = self._tensor(memory, self._input_buffer)
            output_value = self._tensor(memory, self._output_buffer)
            self._input_names = (C.c_char_p * 1)(self._input_name.encode('utf-8'))
            self._output_names = (C.c_char_p * 1)(self._output_name.encode('utf-8'))
            self._inputs, self._outputs = (_P * 1)(input_value), (_P * 1)(output_value)
        except BaseException:
            self.close()
            raise

    def _handle(self, operation, release, *arguments, stack=None):
        handle = _P()
        try:
            status = getattr(self._api, operation)(*arguments, C.byref(handle))
        finally:
            if handle:
                (self._resources if stack is None else stack).callback(getattr(self._api, release), handle)
        self._api.check(status)
        if not handle:
            raise RuntimeError(f'ONNX Runtime returned a null handle from {operation}')
        return handle

    def _scalar(self, operation, kind, *arguments):
        value = kind()
        self._api.check(getattr(self._api, operation)(*arguments, C.byref(value)))
        return value.value

    def _free(self, pointer):
        if pointer:
            self._api.check(self._api.AllocatorFree(self._allocator, pointer))

    def _string(self, operation, *arguments):
        pointer = _P()
        try:
            self._api.check(getattr(self._api, operation)(*arguments, C.byref(pointer)))
            return C.string_at(pointer).decode('utf-8') if pointer else ''
        finally:
            self._free(pointer)

    def _read_node(self, kind):
        name = self._string('SessionGet' + kind + 'Name', self._session, 0, self._allocator)
        with ExitStack() as stack:
            info = self._handle('SessionGet' + kind + 'TypeInfo', 'ReleaseTypeInfo', self._session, 0, stack=stack)
            tensor = _P()  # Borrowed from OrtTypeInfo; must not be separately released.
            self._api.check(self._api.CastTypeInfoToTensorInfo(info, C.byref(tensor)))
            if not tensor or self._scalar('GetTensorElementType', _I, tensor) != 1:
                raise ValueError(f'Model {kind.lower()} must be a static float32 tensor')
            rank = self._scalar('GetDimensionsCount', _S, tensor)
            shape = (C.c_int64 * rank)()
            self._api.check(self._api.GetDimensions(tensor, shape, rank))
            shape = tuple(shape)
            if not shape or any(dimension <= 0 for dimension in shape):
                raise ValueError(f'Model {kind.lower()} requires a positive static shape; got {shape}')
            return name, shape

    def _read_metadata(self):
        with ExitStack() as stack:
            metadata = self._handle('SessionGetModelMetadata', 'ReleaseModelMetadata', self._session, stack=stack)
            keys, count = _PP(), C.c_int64()
            try:
                status = self._api.ModelMetadataGetCustomMetadataMapKeys(
                    metadata, self._allocator, C.byref(keys), C.byref(count))
            finally:
                if keys:
                    stack.callback(self._free, C.cast(keys, _P))
                    for index in range(max(0, count.value)):
                        stack.callback(self._free, keys[index])
            self._api.check(status)
            return {C.string_at(keys[index]).decode('utf-8'):
                    self._string('ModelMetadataLookupCustomMetadataMap', metadata, self._allocator,
                                 C.string_at(keys[index])) for index in range(count.value)}

    def _tensor(self, memory, buffer):
        shape = (C.c_int64 * buffer.ndim)(*buffer.shape)
        return self._handle('CreateTensorWithDataAsOrtValue', 'ReleaseValue', memory,
                            buffer.ctypes.data, buffer.nbytes, shape, buffer.ndim, 1)

    def get_inputs(self):
        return [SimpleNamespace(name=self._input_name, shape=list(self._input_shape), type='tensor(float)')]

    def get_outputs(self):
        return [SimpleNamespace(name=self._output_name, shape=list(self._output_shape), type='tensor(float)')]

    def get_modelmeta(self):
        return SimpleNamespace(custom_metadata_map=dict(self._metadata))

    def get_providers(self):
        return ['DmlExecutionProvider']

    def disable_fallback(self):
        """CPU EP fallback is disabled before native session creation."""

    def _check_open(self):
        if self._closed:
            raise RuntimeError('Native DirectML session is closed')

    def run(self, output_names, input_feed):
        with self._lock:
            self._check_open()
            if not isinstance(input_feed, Mapping) or set(input_feed) != {self._input_name}:
                raise ValueError(f'Expected exactly one input named {self._input_name!r}')
            value = input_feed[self._input_name]
            if not isinstance(value, np.ndarray) or value.dtype != np.float32:
                raise TypeError('Native DirectML input must be a NumPy float32 array')
            if value.shape != self._input_shape:
                raise ValueError(f'Expected input shape {self._input_shape}; got {value.shape}')
            if output_names is not None and list(output_names) not in ([], [self._output_name]):
                raise ValueError(f'Expected output named {self._output_name!r}')
            np.copyto(self._input_buffer, value, casting='no')
            self._api.check(self._api.Run(self._session, None, self._input_names, self._inputs, 1,
                                         self._output_names, 1, self._outputs))
            return [self._output_buffer.copy()]

    def end_profiling(self):
        with self._lock:
            self._check_open()
            if not self._profiling:
                return ''
            if self._profile_path is None:
                self._profile_path = self._string('SessionEndProfiling', self._session, self._allocator)
            return self._profile_path

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._resources.close()
            finally:
                self._input_buffer = self._output_buffer = None

    def __enter__(self):
        with self._lock:
            self._check_open()
        return self

    def __exit__(self, *exception):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
