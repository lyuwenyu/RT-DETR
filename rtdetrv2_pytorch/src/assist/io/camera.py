"""Threaded frame capture that always hands back the newest frame.

A camera delivers ~30 fps while the detector plus depth net take ~75 ms, so
reading frames synchronously would work through a growing backlog and guide the
user using what the camera saw a second ago. Stale guidance is worse than no
guidance, so the reader thread keeps only the most recent frame and drops the
rest. Video files are read in lockstep instead, since there every frame matters
and nothing is real-time.

Network streams -- a phone camera shared over WiFi -- count as live, and they
are the case where dropping stale frames matters most: the stream carries its
own buffer, so a lockstep reader falls progressively further behind real time
rather than merely lagging by a fixed amount.
"""

import threading
import time

import cv2


STREAM_SCHEMES = ('http://', 'https://', 'rtsp://', 'rtmp://', 'udp://', 'tcp://')


def classify_source(source):
    """-> (opencv_source, is_device, is_stream). Live means device or stream."""
    if isinstance(source, int):
        return source, True, False
    text = str(source)
    if text.isdigit():
        return int(text), True, False
    if text.lower().startswith(STREAM_SCHEMES):
        return text, False, True
    return text, False, False


class Camera:
    def __init__(self, source, width=None, height=None):
        self.source, self.is_device, self.is_stream = classify_source(source)
        self.is_live = self.is_device or self.is_stream

        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            if self.is_stream:
                raise RuntimeError(
                    'Cannot open stream: {!r}\n'
                    '  - is the phone on the same WiFi network as this machine?\n'
                    '  - is the streaming app running and showing that exact URL?\n'
                    '  - try opening the URL in a browser on this machine first'
                    .format(source))
            raise RuntimeError('Cannot open video source: {!r}'.format(source))

        if self.is_device:
            if width:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        elif self.is_stream:
            # Keep the decoder's own queue as short as it allows, so the
            # reader thread is discarding at most a frame or two of backlog.
            try:
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = 0 if self.is_live else int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # A stream can report nothing until the first frame lands.
        if self.is_stream and (self.width <= 0 or self.height <= 0):
            ok, probe = self.cap.read()
            if not ok or probe is None:
                raise RuntimeError(
                    'Stream opened but delivered no frames: {!r}'.format(source))
            self.height, self.width = probe.shape[:2]

        self._frame = None
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self.is_live:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
            # Give the first frame a moment to arrive so callers do not have to
            # special-case an empty start.
            deadline = time.time() + 5.0
            while self._frame is None and time.time() < deadline:
                time.sleep(0.01)
        return self

    def read(self):
        """-> (frame_bgr, timestamp) or (None, None) when the source ends."""
        if not self.is_live:
            ok, frame = self.cap.read()
            return (frame, time.time()) if ok else (None, None)

        with self._lock:
            if self._frame is None:
                return None, None
            return self._frame.copy(), self._stamp

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._stamp = time.time()

    def release(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.release()
