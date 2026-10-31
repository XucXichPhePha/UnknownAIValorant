"""Bounded, latest-frame capture from the actual primary display."""
from __future__ import annotations

import re
import time

import numpy as np


def primary_output(info: str) -> tuple[int, int]:
    outputs = re.findall(r"Device\[(\d+)\] Output\[(\d+)\]:[^\n]*Primary:True", info)
    if len(outputs) != 1:
        raise RuntimeError("DXGI could not identify exactly one primary display")
    return tuple(map(int, outputs[0]))


class ScreenCapture:
    def __init__(self, monitor, fps=120, backend="auto", region=None):
        if backend not in ("auto", "dxgi", "mss"):
            raise ValueError("Capture backend must be auto, dxgi or mss")
        self.monitor, self.fps, self.backend = monitor, fps, backend
        self.camera = self.gdi = None
        self.name = ""
        self._last_ticks = None
        self.region = region or (0,0,monitor['width'],monitor['height'])
        x,y,w,h=self.region
        if min(x,y)<0 or min(w,h)<=0 or x+w>monitor['width'] or y+h>monitor['height']:
            raise ValueError('Capture region must be inside the primary display')
        self.frame_time = None

    def __enter__(self):
        if self.backend != "mss":
            try:
                import dxcam
                device, output = primary_output(dxcam.output_info())
                self.camera = dxcam.create(
                    device_idx=device, output_idx=output, output_color="BGRA",
                    max_buffer_len=2, backend="dxgi",
                )
                x,y,w,h=self.region
                self.camera.start(target_fps=self.fps, video_mode=False,region=(x,y,x+w,y+h))
                self.name = "DXGI"
                return self
            except Exception as error:
                if self.camera is not None:
                    self.camera.release()
                    self.camera = None
                if self.backend == "dxgi":
                    raise RuntimeError(f"DXGI capture failed: {error}") from error
                print(f"DXGI unavailable ({error}); using MSS capture.", flush=True)
        import mss
        self.gdi = mss.MSS()
        self.name = "MSS"
        return self

    def grab(self):
        if self.camera is not None:
            if not self.camera.is_capturing:
                raise RuntimeError("DXGI capture stopped. Restart the overlay or use --capture mss.")
            atomic, frame = self._buffered_frame()
            if atomic and frame is None:
                return None
            if not atomic:
                for _ in range(3):
                    ticks = self.camera.latest_frame_ticks
                    if ticks is None or ticks == self._last_ticks:
                        return None
                    # Keep caller-owned memory and a consistent frame timestamp: the
                    # producer can advance between the timestamp read and frame copy.
                    frame = self.camera.grab()
                    if frame is None:
                        return None
                    captured_at=getattr(self.camera,'latest_frame_time',None)
                    if self.camera.latest_frame_ticks == ticks:
                        self._last_ticks = ticks
                        self.frame_time = captured_at if captured_at is not None else time.perf_counter()
                        break
                else:
                    return None
        else:
            x,y,w,h=self.region
            started=time.perf_counter()
            frame = np.asarray(self.gdi.grab({**self.monitor,'left':self.monitor['left']+x,
                'top':self.monitor['top']+y,'width':w,'height':h}))
            self.frame_time = (started+time.perf_counter())/2
        if (frame.shape[:2] != (self.region[3],self.region[2])
            or (self.camera is not None and
                (getattr(self.camera,'width',self.monitor['width']),getattr(self.camera,'height',self.monitor['height']))
                 != (self.monitor['width'],self.monitor['height']))):
            raise RuntimeError("Display resolution changed. Restart the overlay at 1280x882 or 1280x720.")
        return frame

    def _buffered_frame(self):
        runtime = getattr(self.camera, '_DXCamera__capture_runtime', None)
        lock = getattr(self.camera, '_DXCamera__lock', None)
        peek = getattr(runtime, 'peek_latest_with_ticks', None)
        convert = getattr(getattr(self.camera, '_duplicator', None), 'ticks_to_seconds', None)
        if not callable(peek) or not callable(convert) or lock is None:
            return False, None
        if not lock.acquire(False):
            return True, None
        try:
            ticks = getattr(runtime, 'latest_frame_ticks', None)
            if ticks is None or ticks == self._last_ticks:
                return True, None
            packet = peek(copy=True)
            if packet is None:
                return True, None
            frame, ticks = packet
            self.frame_time = convert(ticks)
            self._last_ticks = ticks
            return True, frame
        finally:
            lock.release()

    def __exit__(self, *args):
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        if self.gdi is not None:
            self.gdi.close()
            self.gdi = None
