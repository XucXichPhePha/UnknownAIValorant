import math
import time

from PySide6.QtCore import QObject, QTimer, Qt


class RenderSchedule:
    def __init__(self, fps, *, clock=time.perf_counter):
        if not 1 <= fps <= 240:
            raise ValueError('Render FPS must be between 1 and 240')
        self.period = 1 / fps
        self.clock = clock
        self.deadline = None

    def next_delay(self):
        now = self.clock()
        self.deadline = now + self.period if self.deadline is None else self.deadline + self.period
        if self.deadline <= now:
            self.deadline += (math.floor((now-self.deadline)/self.period)+1)*self.period
        return max(1, math.ceil((self.deadline-now)*1000-1e-7))


class RenderTimer(QObject):
    def __init__(self, parent, fps, callback):
        super().__init__(parent)
        self.schedule = RenderSchedule(fps)
        self.callback = callback
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._tick)
        self.running = False

    def start(self):
        self.running = True
        self.schedule.deadline = None
        self.timer.start(self.schedule.next_delay())

    def stop(self):
        self.running = False
        self.timer.stop()

    def _tick(self):
        try:
            self.callback()
        finally:
            if self.running:
                self.timer.start(self.schedule.next_delay())
