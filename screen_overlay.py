from __future__ import annotations

import ctypes
import sys
import threading
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMenu,
    QMessageBox,
    QSystemTrayIcon,
    QWidget,
)

PT_MODEL_PATH = Path(__file__).resolve().with_name("best.pt")
ENGINE_MODEL_PATH = Path(__file__).resolve().with_name("best.engine")
INFERENCE_FPS = 144
MAX_CAPTURE_SIDE = 640


class DetectionWorker(QThread):
    detections_ready = Signal(list)
    failed = Signal(str)

    def __init__(self, model_path: Path) -> None:
        super().__init__()
        self.model_path = model_path
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        try:
            import mss
            import cv2
            import numpy as np
            import torch
            from ultralytics import YOLO

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Không tìm thấy CUDA. Hãy chạy setup.bat để cài "
                    "PyTorch CUDA cho card NVIDIA."
                )

            device_name = torch.cuda.get_device_name(0)
            inference_backend = (
                "TensorRT"
                if self.model_path.suffix.lower() == ".engine"
                else "PyTorch FP16"
            )
            print(
                f"Đang nhận diện bằng {device_name} "
                f"({inference_backend}, tối đa {INFERENCE_FPS} FPS)."
            )
            model = YOLO(str(self.model_path))
            with mss.MSS() as capture:
                monitor = capture.monitors[1]
                fps_started = time.monotonic()
                frames_processed = 0
                while not self._stop_event.is_set():
                    frame_started = time.monotonic()
                    frame = np.asarray(capture.grab(monitor))[:, :, :3]
                    height, width = frame.shape[:2]
                    scale = min(1.0, MAX_CAPTURE_SIDE / max(width, height))
                    if scale < 1.0:
                        frame = cv2.resize(
                            frame,
                            (round(width * scale), round(height * scale)),
                            interpolation=cv2.INTER_AREA,
                        )
                    predict_args = {
                        "source": frame,
                        "imgsz": 640,
                        "conf": 0.25,
                        "device": 0,
                        "verbose": False,
                    }
                    if self.model_path.suffix.lower() != ".engine":
                        predict_args["quantize"] = 16
                    results = model.predict(**predict_args)

                    current = []
                    for result in results:
                        if result.boxes is None:
                            continue
                        height, width = frame.shape[:2]
                        for box in result.boxes:
                            x1, y1, x2, y2 = (
                                box.xyxy[0].cpu().tolist()
                            )
                            class_id = int(box.cls[0].item())
                            confidence = float(box.conf[0].item())
                            class_name = result.names.get(
                                class_id, str(class_id)
                            )
                            current.append(
                                (
                                    x1 / width,
                                    y1 / height,
                                    x2 / width,
                                    y2 / height,
                                    class_name,
                                    f"{class_name} {confidence:.0%}",
                                )
                            )
                    self.detections_ready.emit(current)
                    frames_processed += 1
                    fps_elapsed = time.monotonic() - fps_started
                    if fps_elapsed >= 5:
                        print(
                            f"Tốc độ nhận diện thực tế: "
                            f"{frames_processed / fps_elapsed:.1f} FPS "
                            f"(mục tiêu {INFERENCE_FPS} FPS)."
                        )
                        fps_started = time.monotonic()
                        frames_processed = 0
                    remaining = 1 / INFERENCE_FPS - (
                        time.monotonic() - frame_started
                    )
                    self._stop_event.wait(max(0, remaining))
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

    def start(self) -> None:
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

    def set_detections(self, detections: list) -> None:
        self._detections = detections
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
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)

        width = self.width()
        height = self.height()
        for x1, y1, x2, y2, class_name, label in self._detections:
            if (
                self._detection_filter != "both"
                and class_name.casefold() != self._detection_filter
            ):
                continue

            left = round(x1 * width)
            top = round(y1 * height)
            right = round(x2 * width)
            bottom = round(y2 * height)
            rect = self.rect().adjusted(left, top, -(width - right), -(height - bottom))

            painter.setPen(QPen(QColor(0, 0, 0, 180), 5))
            painter.drawRect(rect)
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawRect(rect)

            if not self._labels_visible:
                continue

            label_rect = painter.fontMetrics().boundingRect(label).adjusted(
                -6, -3, 6, 3
            )
            label_rect.moveTopLeft(rect.topLeft())
            painter.fillRect(label_rect, QColor(0, 150, 75, 220))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(
                label_rect,
                Qt.AlignmentFlag.AlignCenter,
                label,
            )


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


def main() -> int:
    if sys.platform != "win32":
        print("Ứng dụng này hiện chỉ hỗ trợ Windows.")
        return 1

    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    model_path = (
        ENGINE_MODEL_PATH if ENGINE_MODEL_PATH.is_file() else PT_MODEL_PATH
    )
    if not model_path.is_file():
        QMessageBox.critical(
            None,
            "Không tìm thấy model",
            f"Đặt best.pt cạnh chương trình:\n{PT_MODEL_PATH}",
        )
        return 1

    screen = app.primaryScreen()
    if screen is None:
        QMessageBox.critical(None, "Lỗi", "Không tìm thấy màn hình chính.")
        return 1

    overlay = DetectionOverlay(screen.geometry())
    worker = DetectionWorker(model_path)
    worker.detections_ready.connect(overlay.set_detections)
    console_signals = ConsoleSignals()
    console_signals.labels_visibility_changed.connect(overlay.set_labels_visible)
    console_signals.labels_toggle_requested.connect(overlay.toggle_labels)
    console_signals.detection_filter_changed.connect(
        overlay.set_detection_filter
    )
    console_signals.quit_requested.connect(app.quit)
    console_commands = ConsoleCommandReader(console_signals)

    def show_worker_error(details: str) -> None:
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
    menu.addSeparator()
    menu.addAction(exit_action)
    tray.setContextMenu(menu)

    def toggle_overlay() -> None:
        if overlay.isVisible():
            overlay.hide()
            toggle_action.setText("Hiện lớp phủ")
        else:
            overlay.show()
            toggle_action.setText("Tạm dừng lớp phủ")

    toggle_action.triggered.connect(toggle_overlay)
    tray.activated.connect(
        lambda reason: toggle_overlay()
        if reason == QSystemTrayIcon.ActivationReason.Trigger
        else None
    )
    exit_action.triggered.connect(app.quit)

    overlay.show()
    overlay.exclude_from_capture()
    tray.show()
    print("Đang nhận diện màn hình chính. Dùng biểu tượng khay hệ thống để tạm dừng/thoát.")
    console_commands.start()

    worker.start()
    app.aboutToQuit.connect(worker.stop)
    app.aboutToQuit.connect(worker.wait)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
