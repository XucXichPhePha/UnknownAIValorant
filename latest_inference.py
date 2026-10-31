"""One DirectML runner with one replaceable waiting frame and one result slot."""
from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class InferenceResult:
    frame: object
    timestamp: float
    detections: list
    elapsed: float


class LatestInference:
    def __init__(self,model,fps=120):
        self.model,self.fps=model,fps
        self._condition=threading.Condition()
        self._pending=self._result=self._error=None
        self._stop=False
        self._completed=0
        self._elapsed=0.
        self._thread=threading.Thread(target=self._run,name='onnx-inference',daemon=True)

    @property
    def is_alive(self):return self._thread.is_alive()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self,*args):
        with self._condition:
            self._stop=True
            self._pending=None
            self._condition.notify_all()
        self._thread.join()

    def submit(self,frame,timestamp):
        """Caller-owned frame is shared read-only, never retained in a queue."""
        with self._condition:
            self._pending=(frame,timestamp)
            self._condition.notify_all()

    def take(self):
        with self._condition:
            if self._error is not None:raise RuntimeError(str(self._error)) from self._error
            result,self._result=self._result,None
            return result

    def statistics(self):
        with self._condition:return self._completed,self._elapsed

    def wait_for_completion(self,count,timeout):
        with self._condition:
            return self._condition.wait_for(lambda:self._completed>=count or self._error is not None,timeout)

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda:self._stop or self._pending is not None)
                if self._stop:return
                frame,timestamp=self._pending
                self._pending=None
            started=time.perf_counter()
            try:
                detections=self.model.predict(frame,view='full')
            except Exception as error:
                with self._condition:
                    self._error=error
                    self._condition.notify_all()
                return
            elapsed=time.perf_counter()-started
            with self._condition:
                self._result=InferenceResult(frame,timestamp,detections,elapsed)
                self._completed+=1
                self._elapsed+=elapsed
                self._condition.notify_all()
                if self._stop:return
            # Native high-resolution sleep avoids timed Condition.wait rounding
            # on Windows. Shutdown latency remains bounded by one frame period.
            time.sleep(max(0,1/self.fps-elapsed))
