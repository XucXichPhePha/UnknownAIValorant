from array import array
from bisect import bisect_right
import ctypes
from dataclasses import dataclass
from functools import cached_property
import math
import multiprocessing
import struct
import sys
import threading
import time


@dataclass(frozen=True)
class MouseSnapshot:
    generation: int
    active: bool
    samples: tuple
    available: bool = True

    @cached_property
    def _times(self):
        return tuple(p[0] for p in self.samples)

    @cached_property
    def _gaps(self):
        gaps = [0]
        for previous, current in zip(self._times, self._times[1:]):
            gaps.append(gaps[-1]+(current-previous > .02))
        return tuple(gaps)

    def delta(self, start, end):
        if (not self.active or len(self.samples) < 2 or not math.isfinite(start)
                or not math.isfinite(end) or end < start or start < self.samples[0][0]
                or end > self.samples[-1][0]+.008):
            return None
        times = self._times
        left = max(0, bisect_right(times, start)-1)
        right = min(len(times)-1, bisect_right(times, end))
        if self._gaps[right] != self._gaps[left]:
            return None
        def at(t):
            i = min(len(times)-1, max(0, bisect_right(times, t)-1))
            a = self.samples[i]
            if i == len(times)-1:
                return a[1:]
            b = self.samples[i+1]
            ratio = (t-a[0])/(b[0]-a[0])
            return a[1]+ratio*(b[1]-a[1]), a[2]+ratio*(b[2]-a[2])
        a, b = at(start), at(end)
        return b[0]-a[0], b[1]-a[1]


class MouseHistory:
    capacity = 512

    def __init__(self):
        context = multiprocessing.get_context('spawn')
        self._data = context.Array('d', 7+self.capacity*3)

    def record(self, timestamp, dx, dy, *, active, reset=False):
        if not all(math.isfinite(v) for v in (timestamp, dx, dy)):
            raise ValueError('Mouse motion values must be finite')
        with self._data.get_lock():
            raw = self._data.get_obj()
            state = -1 if active is None else int(bool(active))
            changed = raw[4] != state
            if changed or reset or timestamp <= raw[5] and raw[6]:
                raw[0] = raw[1] = raw[2] = raw[6] = 0
                raw[3] += 1
                dx = dy = 0
            raw[4], raw[5] = state, timestamp
            if not active:
                raw[6] = 0
                return
            index = int(raw[0])
            raw[1] += dx
            raw[2] += dy
            offset = 7+index*3
            raw[offset:offset+3] = timestamp, raw[1], raw[2]
            raw[0] = (index+1) % self.capacity
            raw[6] = min(self.capacity, raw[6]+1)

    def disable(self):
        self.record(time.perf_counter(), 0, 0, active=None)

    def read(self):
        lock = self._data.get_lock()
        if not lock.acquire(False):
            return None
        try:
            raw = self._data.get_obj()
            content = ctypes.string_at(ctypes.addressof(raw), ctypes.sizeof(raw))
        finally:
            lock.release()
        data = array('d')
        data.frombytes(content)
        count, index = int(data[6]), int(data[0])
        samples = tuple(tuple(data[7+((index-count+i) % self.capacity)*3:
                                   10+((index-count+i) % self.capacity)*3]) for i in range(count))
        return MouseSnapshot(int(data[3]), data[4] > 0, samples, data[4] >= 0)


class RawHeader(ctypes.Structure):
    _fields_ = [('kind', ctypes.c_uint32), ('size', ctypes.c_uint32),
                ('device', ctypes.c_void_p), ('param', ctypes.c_size_t)]


def decode_packets(data, count):
    if not 0 <= count <= 2048 or len(data) > 65536:
        raise ValueError('Invalid raw mouse buffer')
    offset = dx = dy = 0
    changed = False
    header_size = ctypes.sizeof(RawHeader)
    alignment = ctypes.sizeof(ctypes.c_void_p)
    for _ in range(count):
        if offset+header_size > len(data):
            raise ValueError('Truncated raw mouse header')
        kind, size = struct.unpack_from('<II', data, offset)
        if size < header_size or offset+size > len(data):
            raise ValueError('Invalid raw mouse packet size')
        if kind == 0:
            if size < header_size+24:
                raise ValueError('Truncated raw mouse packet')
            flags, _, _, x, y, _ = struct.unpack_from('<H2xIIiiI', data, offset+header_size)
            changed |= bool(flags & 4)
            if not flags & 1:
                dx += x
                dy += y
        offset = (offset+size+alignment-1) & ~(alignment-1)
    return dx, dy, changed


class RawMouseReceiver:
    def __init__(self, history):
        self.history = history
        self.error = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is not None:
            return self.error is None
        if sys.platform != 'win32' or ctypes.sizeof(ctypes.c_void_p) != 8:
            self.error = 'Mouse assistance requires 64-bit Windows'
            return False
        self._thread = threading.Thread(target=self._run, name='overlay-mouse', daemon=True)
        self._thread.start()
        if not self._ready.wait(2):
            self.error = 'Mouse receiver did not start'
            self.close()
        return self.error is None

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2)
        self.history.record(time.perf_counter(), 0, 0, active=None if self.error else False)

    def _run(self):
        from ctypes import wintypes as wt
        class Device(ctypes.Structure):
            _fields_ = [('page', wt.USHORT), ('usage', wt.USHORT),
                        ('flags', wt.DWORD), ('window', wt.HWND)]
        user = ctypes.WinDLL('user32', use_last_error=True)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        user.CreateWindowExW.argtypes = [wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                        wt.HWND, wt.HMENU, wt.HINSTANCE, ctypes.c_void_p]
        user.CreateWindowExW.restype = wt.HWND
        user.DestroyWindow.argtypes = [wt.HWND]
        user.RegisterRawInputDevices.argtypes = [ctypes.POINTER(Device), wt.UINT, wt.UINT]
        user.GetRawInputBuffer.argtypes = [ctypes.c_void_p, ctypes.POINTER(wt.UINT), wt.UINT]
        user.GetRawInputBuffer.restype = wt.UINT
        user.PeekMessageW.argtypes = [ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT, wt.UINT]
        user.DispatchMessageW.argtypes = [ctypes.POINTER(wt.MSG)]
        user.DispatchMessageW.restype = ctypes.c_ssize_t
        user.GetForegroundWindow.restype = wt.HWND
        user.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
        kernel.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
        kernel.OpenProcess.restype = wt.HANDLE
        kernel.QueryFullProcessImageNameW.argtypes = [wt.HANDLE, wt.DWORD, wt.LPWSTR, ctypes.POINTER(wt.DWORD)]
        kernel.CloseHandle.argtypes = [wt.HANDLE]
        handle = None
        registered = False
        try:
            handle = user.CreateWindowExW(0, 'STATIC', 'overlay-motion', 0, 0, 0, 0, 0,
                                          wt.HWND(-3), None, None, None)
            if not handle:
                raise ctypes.WinError(ctypes.get_last_error())
            device = Device(1, 2, 0x100, handle)
            if not user.RegisterRawInputDevices(ctypes.byref(device), 1, ctypes.sizeof(Device)):
                raise ctypes.WinError(ctypes.get_last_error())
            registered = True
            buffer = ctypes.create_string_buffer(65536)
            message = wt.MSG()
            previous_pid, active = None, False
            self._ready.set()
            while not self._stop.is_set():
                for _ in range(64):
                    if not (user.PeekMessageW(ctypes.byref(message), None, 0, 0xFE, 1)
                            or user.PeekMessageW(ctypes.byref(message), None, 0x100, 0xFFFF, 1)):
                        break
                    user.DispatchMessageW(ctypes.byref(message))
                pid = wt.DWORD()
                user.GetWindowThreadProcessId(user.GetForegroundWindow(), ctypes.byref(pid))
                if pid.value != previous_pid:
                    previous_pid, active = pid.value, False
                    process = kernel.OpenProcess(0x1000, False, pid.value)
                    if process:
                        try:
                            name, length = ctypes.create_unicode_buffer(32768), wt.DWORD(32768)
                            if kernel.QueryFullProcessImageNameW(process, 0, name, ctypes.byref(length)):
                                active = name.value.rsplit('\\', 1)[-1].casefold() == 'valorant-win64-shipping.exe'
                        finally:
                            kernel.CloseHandle(process)
                dx = dy = 0
                changed = False
                for _ in range(8):
                    size = wt.UINT(len(buffer))
                    count = user.GetRawInputBuffer(buffer, ctypes.byref(size), ctypes.sizeof(RawHeader))
                    if count == 0xFFFFFFFF:
                        raise ctypes.WinError(ctypes.get_last_error())
                    if not count:
                        break
                    if active:
                        x, y, reset = decode_packets(buffer.raw, count)
                        dx += x
                        dy += y
                        changed |= reset
                self.history.record(time.perf_counter(), dx, dy, active=active, reset=changed)
                time.sleep(.004)
        except Exception as error:
            self.error = str(error)
        finally:
            if registered:
                device = Device(1, 2, 1, None)
                user.RegisterRawInputDevices(ctypes.byref(device), 1, ctypes.sizeof(Device))
            if handle:
                user.DestroyWindow(handle)
            self.history.record(time.perf_counter(), 0, 0, active=None if self.error else False)
            self._ready.set()
