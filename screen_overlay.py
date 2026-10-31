from __future__ import annotations

import argparse
import ctypes
import math
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from queue import Empty

from PySide6.QtCore import QObject, QRect, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QColor, QFont, QFontMetrics, QIcon, QPainter, QPen, QPixmap, QStaticText, QTransform
from PySide6.QtWidgets import (
    QApplication,
    QMenu,
    QMessageBox,
    QSystemTrayIcon,
    QWidget,
)

ONNX_MODEL_PATH = Path(__file__).resolve().with_name("best.onnx")
INFERENCE_FPS = 60


def resolve_model_options(project,model=None,view=None,registry=None):
    project=Path(project)
    if model is not None:
        return Path(model).expanduser().resolve(),view or 'fast'
    if registry is None:
        from model_registry import ModelRegistry
        registry=ModelRegistry(project)
    selected=registry.active()
    if selected is not None:
        if view == 'compact':
            return Path(selected['path']).expanduser().resolve(), view
        if view is not None:
            sibling=registry.for_view(selected['id'],view)
            if sibling is None:
                raise ValueError(f'The selected model has no {view} variant. Choose an available view in the board or pass --model explicitly.')
            return Path(sibling['path']).expanduser().resolve(),view
        return Path(selected['path']).expanduser().resolve(),view or selected['view']
    view=view or 'fast'
    names={'fast':'best-focus-192x128.onnx','tight':'best-focus-256x192.onnx'}
    return (project/names.get(view,'best.onnx')).resolve(),view


def overlay_geometry(screen, native_shape, region):
    height,width=native_shape
    x,y,w,h=region
    sx,sy=screen.width()/width,screen.height()/height
    left,top=math.floor(x*sx),math.floor(y*sy)
    right,bottom=math.ceil((x+w)*sx),math.ceil((y+h)*sy)
    return QRect(screen.x()+left,screen.y()+top,right-left,bottom-top)


def localize_overlay_detections(detections,screen,geometry):
    left,top=geometry.x()-screen.x(),geometry.y()-screen.y()
    return [((x1*screen.width()-left)/geometry.width(),(y1*screen.height()-top)/geometry.height(),
             (x2*screen.width()-left)/geometry.width(),(y2*screen.height()-top)/geometry.height(),name,label)
            for x1,y1,x2,y2,name,label in detections]


class DetectionWorker(QThread):
    detections_ready = Signal(list)
    status_ready = Signal(str)
    failed = Signal(str)

    def __init__(self, model_path: Path, fps=INFERENCE_FPS, device_id=0, capture_backend="auto", view="focus", tracking=True, inference_fps=60, isolated=True, verbose=False, runtime='native', recordable_overlay=False, motion_assistance=False, mouse_history=None) -> None:
        super().__init__()
        self.model_path = model_path
        self.fps = fps
        self.device_id = device_id
        self.capture_backend = capture_backend
        self.view = view
        self.tracking = tracking
        self.inference_fps = inference_fps
        self.isolated=isolated
        self.verbose=verbose
        self.runtime=runtime
        self.recordable_overlay=recordable_overlay
        self.motion_assistance = motion_assistance and tracking
        self.mouse_history = mouse_history
        self.motion_metadata = None
        self._render_updates=None
        self._engine=None
        self.capture_timestamp=self.model_timestamp=None
        self._stop_event = threading.Event()
        self._display_lock = threading.Lock()
        self._display_update = None

    def stop(self) -> None:
        self._stop_event.set()
        engine=self._engine
        if engine is not None:engine.stop()

    def take_detections(self):
        packet = self.take_display()
        return packet[0] if packet is not None else None

    def take_display(self):
        with self._display_lock:
            current,self._display_update=self._display_update,None
            return current

    def send_rendered(self, regions):
        engine = self._engine
        return engine is not None and engine.send_rendered(regions)

    def _publish(self,current):
        with self._display_lock:self._display_update=(current, self.motion_metadata)
        # Retained for external diagnostics; the GUI consumes only the latest
        # state on its own timer, so old queued signals cannot build latency.
        self.detections_ready.emit(current)

    def run(self) -> None:
        if self.isolated:
            try:
                from overlay_process import ProcessEngine
                config=dict(model_path=str(self.model_path),fps=self.fps,device_id=self.device_id,
                    capture_backend=self.capture_backend,view=self.view,tracking=self.tracking,
                    inference_fps=self.inference_fps,verbose=self.verbose,runtime=self.runtime,
                    recordable_overlay=self.recordable_overlay, motion_assistance=self.motion_assistance,
                    mouse_history=self.mouse_history)
                last_status=None
                with ProcessEngine(config) as engine:
                    self._engine=engine
                    while not self._stop_event.is_set():
                        packet=engine.poll()
                        if packet is None:continue
                        status=packet.get('status')
                        if status and status!=last_status:
                            self.status_ready.emit(status)
                            last_status=status
                        if packet.get('detections') is not None:
                            self.capture_timestamp=packet.get('capture_timestamp')
                            self.model_timestamp=packet.get('model_timestamp')
                            self.motion_metadata=packet.get('motion')
                            self._publish(packet['detections'])
            except Exception:
                self.failed.emit(traceback.format_exc())
            finally:self._engine=None
            return
        try:
            import mss
            from directml_backend import capture_region, frame_delay, remap_detections, validate_capture_size
            from overlay_backend import create_detector, model_banner
            from latest_inference import LatestInference
            from visual_tracking import PixelTracker
            from screen_capture import ScreenCapture
            with mss.MSS() as info:
                monitor = info.primary_monitor
            validate_capture_size(monitor["width"], monitor["height"])
            model = create_detector(self.model_path, self.device_id, runtime=self.runtime)
            backend = model.backend_label
            print(model_banner(self.model_path, model), flush=True)
            region=capture_region(monitor['width'],monitor['height'],self.view)
            tracker=PixelTracker(assistance=self.motion_assistance, mouse_history=self.mouse_history)
            cleaner = None
            if self.recordable_overlay:
                from overlay_feedback import OverlayPixelCleaner
                cleaner = OverlayPixelCleaner(monitor['width'], monitor['height'], region)
            with ScreenCapture(monitor, self.fps, self.capture_backend,region=region) as capture, \
                    LatestInference(model,self.inference_fps) as inference:
                if self.verbose:
                    print(
                        f"{backend}, native capture "
                        f"{monitor['width']}x{monitor['height']}, model {model.size}px, "
                        f"{capture.name}, view {self.view}, target {self.fps} FPS. "
                        f"Input {model.input_size[0]}x{model.input_size[1]}, pixel tracking {self.tracking}.", flush=True,
                    )
                fps_started = time.perf_counter()
                frames_processed = 0
                fresh_frames = 0
                capture_total = tracking_total = age_total = 0.0
                last_frame=None
                timestamp=0.
                previous_completed=0
                previous_processing=0.
                model_ready=False
                pending_result=None
                first_status = True
                self.status_ready.emit(
                    f"{backend} / {capture.name} | {monitor['width']}x{monitor['height']} | {self.view} | target {self.fps} FPS\n"
                    "Waiting for first captured frame..."
                )
                while not self._stop_event.is_set():
                    frame_started = time.perf_counter()
                    if cleaner is not None and self._render_updates is not None:
                        for _ in range(8):
                            try:
                                cleaner.set_regions(self._render_updates.get_nowait())
                            except Empty:
                                break
                    frame = capture.grab()
                    captured_at = time.perf_counter()
                    result=inference.take()
                    if result is not None:pending_result=result
                    if frame is not None:
                        if cleaner is not None:
                            frame = cleaner.clean(frame)
                        last_frame=frame
                        timestamp=getattr(capture,'frame_time',None) or captured_at
                        self.capture_timestamp=timestamp
                        inference.submit(frame,timestamp)
                    if last_frame is None or (frame is None and (model_ready or pending_result is None)):
                        # Python3.12 sleep uses a high-resolution Windows timer.
                        # Event.wait's millisecond timeout can round to a whole
                        # desktop timer tick and halve the120Hz capture cadence.
                        time.sleep(.001)
                        continue
                    result,pending_result=pending_result,None
                    kwargs={}
                    if result is not None:
                        model_ready=True
                        self.model_timestamp=result.timestamp
                        kwargs=dict(detections=result.detections,source_frame=result.frame,source_time=result.timestamp)
                    if self.tracking:
                        current=tracker.update(last_frame,timestamp,**kwargs)
                    elif result is not None:
                        current=result.detections
                    else:
                        if capture.name=='MSS':time.sleep(frame_delay(self.fps,time.perf_counter()-frame_started))
                        continue
                    tracked_at = time.perf_counter()
                    self.motion_metadata = tracker.motion_state
                    if self.motion_metadata is not None:
                        self.motion_metadata.update(display_size=(monitor['width'], monitor['height']), region=region)
                    current=remap_detections(current,region,(monitor['height'],monitor['width']))
                    if model_ready:self._publish(current)
                    frames_processed += 1
                    fresh_frames += frame is not None
                    capture_total += captured_at - frame_started
                    tracking_total += tracked_at - captured_at
                    age_total += max(0,tracked_at-(timestamp if self.tracking else result.timestamp))
                    fps_elapsed = time.perf_counter() - fps_started
                    if model_ready and (fps_elapsed >= 1 or first_status):
                        completed,processing=inference.statistics()
                        net_frames=completed-previous_completed
                        net_time=processing-previous_processing
                        message = (
                            f"{backend} / {capture.name} | {monitor['width']}x{monitor['height']} | {self.view} | target {self.fps} FPS\n"
                            f"{fresh_frames / fps_elapsed:.0f} view / {net_frames / fps_elapsed:.0f} infer FPS | {len(current)} marks | "
                            f"cap {capture_total * 1000 / frames_processed:.1f} net {net_time * 1000 / max(1,net_frames):.1f} "
                            f"track {tracking_total * 1000 / frames_processed:.1f}ms | age {age_total * 1000 / frames_processed:.0f}ms"
                        )
                        self.status_ready.emit(message)
                        if self.verbose:
                            print(message.replace("\n", " | "), flush=True)
                        fps_started = time.perf_counter()
                        frames_processed = 0
                        fresh_frames = 0
                        capture_total = tracking_total = age_total = 0.0
                        previous_completed,previous_processing=completed,processing
                        first_status = False
                    # DXGI already paces its producer and skips repeated frames.
                    # A second timed wait here reduces throughput and adds age.
                    if capture.name=='MSS':
                        time.sleep(frame_delay(self.fps,time.perf_counter()-frame_started))
        except Exception:
            self.failed.emit(traceback.format_exc())


class ConsoleSignals(QObject):
    labels_visibility_changed = Signal(bool)
    labels_toggle_requested = Signal()
    detection_filter_changed = Signal(str)
    quit_requested = Signal()


class ConsoleCommandReader:
    def __init__(self, signals: ConsoleSignals) -> None:
        self.signals = signals
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        threading.Thread(target=self._read_commands, daemon=True).start()

    def _read_commands(self) -> None:
        print(
            "Lệnh: text on/off/toggle | show head/body/both | help | quit",
            flush=True,
        )
        while True:
            try:
                command = input("CMD> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return

            if command in {"text on", "chu bat"}:
                self.signals.labels_visibility_changed.emit(True)
                print("Đã bật chữ.", flush=True)
            elif command in {"text off", "chu tat"}:
                self.signals.labels_visibility_changed.emit(False)
                print("Đã tắt chữ; khung vẫn hiển thị.", flush=True)
            elif command in {"text toggle", "chu doi"}:
                self.signals.labels_toggle_requested.emit()
                print("Đã đảo trạng thái chữ.", flush=True)
            elif command in {"show head", "chi head"}:
                self.signals.detection_filter_changed.emit("head")
                print("Chỉ hiển thị Head.", flush=True)
            elif command in {"show body", "chi body"}:
                self.signals.detection_filter_changed.emit("body")
                print("Chỉ hiển thị Body.", flush=True)
            elif command in {"show both", "show all", "hien tat ca"}:
                self.signals.detection_filter_changed.emit("both")
                print("Hiển thị cả Head và Body.", flush=True)
            elif command in {"help", "trogiup"}:
                print(
                    "text on / chu bat: bật chữ; text off / chu tat: tắt chữ; "
                    "text toggle / chu doi: đảo trạng thái; "
                    "show head / chi head: chỉ hiện Head; "
                    "show body / chi body: chỉ hiện Body; "
                    "show both / hien tat ca: hiện cả hai; "
                    "quit / thoat: thoát.",
                    flush=True,
                )
            elif command in {"quit", "thoat"}:
                self.signals.quit_requested.emit()
                return
            else:
                print("Lệnh không hợp lệ. Gõ help để xem các lệnh.", flush=True)


class DetectionOverlay(QWidget):
    def __init__(self, screen_geometry) -> None:
        super().__init__()
        self._detections = []
        self._labels_visible = True
        self._detection_filter = "both"
        self._status_text = "Starting DirectML..."
        self._test_box = False
        self._color_key_enabled = False
        self._show_status = True
        self._status_font=QFont()
        self._status_font.setPointSize(10)
        self._marker_font=QFont()
        self._marker_font.setPixelSize(10)
        self._marker_metrics=QFontMetrics(self._marker_font)
        self._label_sizes={}
        self._static_text={}
        self._marker_key=None
        self._markers=[]
        self._marker_bounds=QRect()
        self.setAccessibleName(self._status_text)
        self.setGeometry(screen_geometry)
        self.setWindowTitle("Screen Detection Overlay")
        self.setWindowFlags(
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

    def enable_capture_transparency(self) -> None:
        """Use redirected color-key transparency on Windows so capture exclusion works.

        Qt's per-pixel alpha window fails SetWindowDisplayAffinity with error 8
        on this PC. The color-key window keeps GDI redirection and click-through.
        """
        if QApplication.platformName() != "windows":
            return
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, False)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        get_style = user32.GetWindowLongPtrW
        get_style.argtypes = (ctypes.c_void_p, ctypes.c_int)
        get_style.restype = ctypes.c_ssize_t
        set_style = user32.SetWindowLongPtrW
        set_style.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t)
        set_style.restype = ctypes.c_ssize_t
        set_layer = user32.SetLayeredWindowAttributes
        set_layer.argtypes = (ctypes.c_void_p, ctypes.c_uint, ctypes.c_ubyte, ctypes.c_uint)
        set_layer.restype = ctypes.c_bool
        handle = ctypes.c_void_p(int(self.winId()))
        ctypes.set_last_error(0)
        previous = set_style(handle, -20, get_style(handle, -20) | 0x00080000)
        if not previous and ctypes.get_last_error():
            raise ctypes.WinError(ctypes.get_last_error())
        if not set_layer(handle, 0x00010101, 255, 0x00000001):
            raise ctypes.WinError(ctypes.get_last_error())
        self._color_key_enabled = True

    def set_detections(self, detections: list) -> None:
        if detections == self._detections:
            return
        self._marker_items()
        dirty=self._marker_bounds
        self._detections = detections
        self._marker_key=None
        self._marker_items()
        dirty=dirty.united(self._marker_bounds)
        if not dirty.isEmpty():
            self.update(dirty.adjusted(-3,-3,3,3))

    def _sync_marker_font(self):
        size=max(6,round(10/self.devicePixelRatioF()))
        strategy=QFont.StyleStrategy.NoAntialias if self._color_key_enabled else QFont.StyleStrategy.PreferDefault
        if self._marker_font.pixelSize()!=size or self._marker_font.styleStrategy()!=strategy:
            self._marker_font.setPixelSize(size)
            self._marker_font.setStyleStrategy(strategy)
            self._marker_metrics=QFontMetrics(self._marker_font)
            self._label_sizes.clear()
            self._static_text.clear()

    def _marker_items(self):
        self._sync_marker_font()
        key=(self.width(),self.height(),self.devicePixelRatioF(),self._labels_visible,
             self._detection_filter,self._marker_font.pixelSize(),self._marker_font.styleStrategy())
        if self._marker_key != key:
            self._markers=[]
            self._marker_bounds=QRect()
            for detection in self._detections:
                name=detection[4].casefold()
                if self._detection_filter != 'both' and name != self._detection_filter:
                    continue
                rect,label,text=self._marker_geometry(detection)
                self._markers.append((rect,label,text,0 if name=='head' else 1))
                self._marker_bounds=self._marker_bounds.united(rect.united(label))
            self._marker_key=key
        return self._markers

    def _prepared_text(self, text):
        if text not in self._static_text:
            item=QStaticText(text)
            item.setTextFormat(Qt.TextFormat.PlainText)
            item.setPerformanceHint(QStaticText.PerformanceHint.AggressiveCaching)
            item.prepare(QTransform(),self._marker_font)
            self._static_text[text]=item
        return self._static_text[text]

    def _marker_geometry(self,detection):
        x1,y1,x2,y2,name,label=detection
        left,top=round(x1*self.width()),round(y1*self.height())
        right,bottom=round(x2*self.width()),round(y2*self.height())
        rect=QRect(left,top,max(1,right-left),max(1,bottom-top))
        text=label.rsplit(' ',1)[-1] if label and label.endswith('%') else ''
        label_rect=QRect()
        if self._labels_visible and text:
            if text not in self._label_sizes:
                self._label_sizes[text]=(self._marker_metrics.horizontalAdvance(text)+2,self._marker_metrics.height())
            w,h=self._label_sizes[text]
            gap=max(2,round(2/self.devicePixelRatioF()))
            y=top-gap-h if name.casefold()=='head' else bottom+gap
            label_rect=QRect(round((left+right-w)/2),y,w,h)
        return rect,label_rect,text

    @Slot(str)
    def set_runtime_status(self, text: str) -> None:
        self._status_text = text
        self.setAccessibleName(text)
        self.update(QRect(10,10,min(664,self.width()-20),58))

    def set_test_box(self, enabled: bool) -> None:
        self._test_box = enabled
        self.update()

    @Slot(bool)
    def set_labels_visible(self, visible: bool) -> None:
        self._labels_visible = visible
        self.update()

    @Slot()
    def toggle_labels(self) -> None:
        self.set_labels_visible(not self._labels_visible)

    @Slot(str)
    def set_detection_filter(self, detection_filter: str) -> None:
        if detection_filter not in {"head", "body", "both"}:
            raise ValueError(f"Unsupported detection filter: {detection_filter}")
        self._detection_filter = detection_filter
        self.update()

    def exclude_from_capture(self) -> bool:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        set_affinity = user32.SetWindowDisplayAffinity
        set_affinity.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        set_affinity.restype = ctypes.c_bool
        if set_affinity(ctypes.c_void_p(int(self.winId())), 0x00000011):
            return True
        error = ctypes.get_last_error()
        print(
            "Cảnh báo: Windows không loại được lớp phủ khỏi ảnh chụp màn hình "
            f"(lỗi {error}); kết quả nhận diện có thể bị ảnh hưởng."
        )
        return False

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        if self._color_key_enabled:
            painter.fillRect(self.rect(), QColor(1, 1, 1))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self._status_font)

        width = self.width()
        height = self.height()
        # A visible heartbeat distinguishes an idle detector from a stopped app.
        if self._show_status:
            painter.fillRect(QRect(12, 12, min(660, width - 24), 54), QColor(12, 18, 24, 220))
            painter.setPen(QColor(80, 240, 190))
            painter.drawText(QRect(22, 17, width - 44, 46), Qt.AlignmentFlag.AlignLeft, self._status_text)
        if self._test_box:
            painter.setPen(QPen(QColor(0, 255, 255), 3))
            test_rect = QRect(round(width * .4), round(height * .4), round(width * .2), round(height * .2))
            painter.drawRect(test_rect)
            painter.drawText(test_rect.adjusted(4, 4, -4, -4), Qt.AlignmentFlag.AlignTop,
                             "OVERLAY DISPLAY TEST")
        # Color-key transparency removes one exact RGB value, not edge alpha.
        # Antialiasing onto that opaque key would leave a near-black fringe.
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, not self._color_key_enabled)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, not self._color_key_enabled)
        self._sync_marker_font()
        painter.setFont(self._marker_font)
        markers=self._marker_items()
        for color_id,color in enumerate((QColor(178,239,188),QColor(128,185,252))):
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(color,2/self.devicePixelRatioF()))
            boxes=[rect for rect,_,_,item_color in markers if item_color==color_id]
            if boxes:
                painter.drawRects(boxes)
            for _,label_rect,text,item_color in markers:
                if item_color!=color_id or label_rect.isEmpty():
                    continue
                item=self._prepared_text(text)
                size=item.size()
                painter.drawStaticText(label_rect.x()+(label_rect.width()-size.width())/2,
                                       label_rect.y()+(label_rect.height()-size.height())/2,item)


def make_tray_icon() -> QIcon:
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(0, 210, 100))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(5, 5, 22, 22)
    painter.end()
    return QIcon(pixmap)


def render_interval(fps):
    return max(1, math.ceil(1000 / fps))


def rendered_regions(overlay, screen):
    if not overlay.isVisible():
        return []
    regions = []
    geometry = overlay.geometry()
    for box,label,_,color_id in overlay._marker_items():
        for rect, outline in ((box, True), (label, False)):
            if rect.isEmpty():
                continue
            x = geometry.x() - screen.x() + rect.x()
            y = geometry.y() - screen.y() + rect.y()
            regions.append((x/screen.width(), y/screen.height(),
                            (x+rect.width())/screen.width(), (y+rect.height())/screen.height(),
                            color_id, outline))
    return regions


def show_overlays(overlay, status_overlay, anti_recording):
    for window in (overlay, status_overlay):
        if window is not None:
            window.show()
            if anti_recording:
                window.exclude_from_capture()


def parse_options(argv=None, *, project=None):
    from overlay_config import load_config
    project = Path(project) if project is not None else ONNX_MODEL_PATH.parent
    parser = argparse.ArgumentParser(description="DirectML / Ultralytics screen detection overlay")
    parser.add_argument('--config', type=Path, help='Overlay YAML settings; defaults to config.yml beside the launcher')
    parser.add_argument("--fps", '--view-fps', type=int, choices=range(1, 241), default=None, metavar="1..240",
                        help='Capture/tracking FPS limit; overrides max_view_fps')
    parser.add_argument('--render-fps', type=int, choices=range(1, 241), default=None, metavar='1..240',
                        help='Overlay drawing limit; overrides max_render_fps')
    parser.add_argument("--inference-fps",type=int,choices=range(1,241),default=None,metavar="1..240",
                        help="Neural FPS limit; defaults to the view FPS limit")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--capture", choices=("auto", "dxgi", "mss"), default="auto")
    parser.add_argument("--view", choices=("tight", "fast", "compact", "focus", "full"), default=None,
                        help="Native center view: compact 256x192, fast 384x256, tight 512x384, focus 640x448, or full display")
    parser.add_argument("--model", type=Path,
                        help="Use an ONNX model with DirectML or a PT checkpoint with Ultralytics")
    parser.add_argument("--no-prediction", action="store_true",
                        help="Disable pixel tracking and display only sparse model updates")
    parser.add_argument("--test-overlay", action="store_true", help="Show a labeled display-test box for five seconds")
    parser.add_argument("--verbose", action="store_true", help="Print startup and FPS diagnostics to the console")
    parser.add_argument("--python-runtime",action="store_true",help="Use the previous Python DirectML path for comparison")
    args = parser.parse_args(argv)
    if args.device_id < 0:
        parser.error("--device-id must be nonnegative")
    try:
        config = load_config(args.config or project / 'config.yml', required=args.config is not None)
    except ValueError as error:
        parser.error(str(error))
    args.fps = args.fps if args.fps is not None else config.max_view_fps
    args.inference_fps = args.inference_fps if args.inference_fps is not None else args.fps
    args.render_fps = args.render_fps if args.render_fps is not None else config.max_render_fps
    args.model = args.model if args.model is not None else config.model_path
    args.anti_recording = config.anti_recording
    args.show_debug_info = config.show_debug_info
    args.view = args.view if args.view is not None else config.capture_view
    args.motion_assistance = config.motion_assistance and not args.no_prediction
    return args


def main(argv=None) -> int:
    if sys.platform != "win32":
        print("This overlay supports Windows only.")
        return 1
    sys.setswitchinterval(.001)
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_options(argv)
    try:
        model_path,args.view=resolve_model_options(ONNX_MODEL_PATH.parent,args.model,args.view)
    except (ValueError,OSError) as error:
        print(f'Could not select the overlay model: {error}',file=sys.stderr)
        return 1

    # Initialize Qt's DPI context before MSS configures capture awareness.
    app = QApplication([sys.argv[0]])
    app.setQuitOnLastWindowClosed(False)

    import mss
    from directml_backend import validate_capture_size
    try:
        with mss.MSS() as capture:
            primary = capture.primary_monitor
            validate_capture_size(primary["width"], primary["height"])
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1

    if not model_path.is_file():
        QMessageBox.critical(
            None,
            "Không tìm thấy model",
            f"Run build_focus_models.bat for tight/fast views, build_onnx.bat for the original, or select --model.\nMissing model:\n{model_path}",
        )
        return 1

    screen = app.primaryScreen()
    if screen is None:
        QMessageBox.critical(None, "Lỗi", "Không tìm thấy màn hình chính.")
        return 1

    from directml_backend import capture_region
    region=capture_region(primary['width'],primary['height'],args.view)
    x,y,w,h=region
    # Leave room for confidence labels above/below boxes at the capture boundary.
    geometry=overlay_geometry(screen.geometry(),(primary['height'],primary['width']),region)
    overlay = DetectionOverlay(geometry.adjusted(-16,-16,16,16).intersected(screen.geometry()))
    overlay._show_status=False
    status_overlay = DetectionOverlay(QRect(screen.geometry().x(),screen.geometry().y(),700,80)) if args.show_debug_info else None
    try:
        overlay.enable_capture_transparency()
        if status_overlay is not None:
            status_overlay.enable_capture_transparency()
    except OSError as error:
        print(f"Could not initialize overlay transparency: {error}", file=sys.stderr)
        return 1
    mouse_history = mouse_receiver = None
    if args.motion_assistance:
        from mouse_motion import MouseHistory, RawMouseReceiver
        mouse_history = MouseHistory()
        mouse_receiver = RawMouseReceiver(mouse_history)
        if not mouse_receiver.start():
            print(f'Mouse assistance unavailable: {mouse_receiver.error}. Using image tracking.', file=sys.stderr)
            mouse_receiver.close()
            mouse_history = None
    worker = DetectionWorker(model_path, args.fps, args.device_id, args.capture, args.view,
                             tracking=not args.no_prediction,
                             inference_fps=args.inference_fps,
                             verbose=args.verbose,runtime='python' if args.python_runtime else 'native',
                             recordable_overlay=not args.anti_recording,
                             motion_assistance=args.motion_assistance, mouse_history=mouse_history)
    last_rendered = None
    last_display = None
    mouse_failure_reported = False
    def refresh_detections():
        nonlocal last_rendered, last_display, mouse_failure_reported
        packet=worker.take_display()
        if packet is not None:
            last_display=packet
        if last_display is not None:
            from motion_render import project_detections
            current=project_detections(*last_display, time.perf_counter(), mouse_history)
            current=localize_overlay_detections(current,screen.geometry(),overlay.geometry())
            overlay.set_detections(current)
        if mouse_receiver is not None and mouse_receiver.error and not mouse_failure_reported:
            print(f'Mouse assistance stopped: {mouse_receiver.error}. Using image tracking.', file=sys.stderr)
            mouse_failure_reported = True
        if not args.anti_recording:
            regions = rendered_regions(overlay, screen.geometry())
            if regions != last_rendered and worker.send_rendered(regions):
                last_rendered = regions
    from overlay_render import RenderTimer
    display_timer=RenderTimer(overlay,args.render_fps,refresh_detections)
    display_timer.start()
    if status_overlay is not None:
        worker.status_ready.connect(status_overlay.set_runtime_status)
    console_signals = ConsoleSignals()
    console_signals.labels_visibility_changed.connect(overlay.set_labels_visible)
    console_signals.labels_toggle_requested.connect(overlay.toggle_labels)
    console_signals.detection_filter_changed.connect(
        overlay.set_detection_filter
    )
    console_signals.quit_requested.connect(app.quit)
    console_commands = ConsoleCommandReader(console_signals)
    worker.status_ready.connect(lambda text: console_commands.start())

    worker_error = False

    def show_worker_error(details: str) -> None:
        nonlocal worker_error
        worker_error = True
        print(details, file=sys.stderr)
        QMessageBox.critical(
            None,
            "Không thể chạy nhận diện",
            "Đã xảy ra lỗi khi nạp model hoặc nhận diện màn hình. "
            "Xem thông tin chi tiết trong cửa sổ console.",
        )
        app.quit()

    worker.failed.connect(show_worker_error)

    tray = QSystemTrayIcon(make_tray_icon(), app)
    tray.setToolTip("Nhận diện màn hình")
    menu = QMenu()
    toggle_action = QAction("Tạm dừng lớp phủ", menu)
    exit_action = QAction("Thoát", menu)
    menu.addAction(toggle_action)
    test_action = QAction("Test overlay (5 seconds)", menu)
    menu.addAction(test_action)
    menu.addSeparator()
    menu.addAction(exit_action)
    tray.setContextMenu(menu)

    def test_overlay() -> None:
        overlay.show()
        overlay.set_test_box(True)
        QTimer.singleShot(5000, lambda: overlay.set_test_box(False))

    test_action.triggered.connect(test_overlay)

    def toggle_overlay() -> None:
        if overlay.isVisible():
            overlay.hide()
            if status_overlay is not None:
                status_overlay.hide()
            toggle_action.setText("Hiện lớp phủ")
        else:
            show_overlays(overlay, status_overlay, args.anti_recording)
            toggle_action.setText("Tạm dừng lớp phủ")

    toggle_action.triggered.connect(toggle_overlay)
    tray.activated.connect(
        lambda reason: toggle_overlay()
        if reason == QSystemTrayIcon.ActivationReason.Trigger
        else None
    )
    exit_action.triggered.connect(app.quit)

    show_overlays(overlay, status_overlay, args.anti_recording)
    if args.test_overlay:
        test_overlay()
    tray.show()
    if args.verbose:
        print("Đang nhận diện màn hình chính. Dùng biểu tượng khay hệ thống để tạm dừng/thoát.")
    worker.start()
    app.aboutToQuit.connect(worker.stop)
    app.aboutToQuit.connect(worker.wait)
    if mouse_receiver is not None:
        app.aboutToQuit.connect(mouse_receiver.close)
    if stop_file:=os.environ.get('UNKNOWNAI_STOP_FILE'):
        stop_path=Path(stop_file)
        stop_timer=QTimer(app)
        stop_timer.timeout.connect(lambda:app.quit() if stop_path.exists() else None)
        stop_timer.start(250)
    result = app.exec()
    return 1 if worker_error else result


if __name__ == "__main__":
    raise SystemExit(main())
