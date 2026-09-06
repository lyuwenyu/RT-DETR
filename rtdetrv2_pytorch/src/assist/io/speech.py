"""Text-to-speech on a worker thread, with a single-slot preemptable mailbox.

pyttsx3's `runAndWait` blocks until the phrase finishes, so it cannot live on
the guidance loop without stalling detection for the length of every sentence.

The mailbox holds exactly one pending phrase, never a queue. That is a design
decision, not a simplification: warnings go stale in about the time it takes to
say them, and a queue would faithfully deliver a backlog describing a scene the
user has already walked through. If something more urgent arrives while a
phrase is waiting, it replaces it. If something less urgent arrives, it is
dropped. Either way the user only ever hears the current state of the world.
"""

import queue
import threading


class Speaker:
    def __init__(self, rate=190, voice=None, enabled=True):
        self.rate = rate
        self.voice = voice
        self.enabled = enabled

        self._pending = None          # (priority, text)
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self.spoken = []              # history, useful for logging and tests

    # -- control ------------------------------------------------------------

    def say(self, text, priority=0):
        """Offer a phrase. Kept only if nothing more urgent is already waiting."""
        if not text:
            return False
        with self._lock:
            if self._pending is not None and self._pending[0] > priority:
                return False
            self._pending = (priority, text)
        self._wake.set()
        return True

    def start(self):
        if not self.enabled:
            return self
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -- worker -------------------------------------------------------------

    def _run(self):
        engine = self._build_engine()
        if engine is None:
            return

        while not self._stop.is_set():
            self._wake.wait(timeout=0.2)
            self._wake.clear()

            with self._lock:
                item, self._pending = self._pending, None
            if item is None:
                continue

            text = item[1]
            self.spoken.append(text)
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception:
                # A failing speech engine must never take the guidance loop
                # down with it -- the beep channel still works without it.
                pass

    def _build_engine(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            if self.voice:
                for v in engine.getProperty('voices'):
                    if self.voice.lower() in v.name.lower():
                        engine.setProperty('voice', v.id)
                        break
            return engine
        except Exception:
            return None


class NullSpeaker:
    """Records what would have been said. Used by --no-audio and by tests."""

    def __init__(self):
        self.spoken = []

    def say(self, text, priority=0):
        if text:
            self.spoken.append(text)
        return True

    def start(self):
        return self

    def stop(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass
