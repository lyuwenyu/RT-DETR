"""Continuous stereo beep synthesis on a real-time audio callback.

The beep channel exists because speech is too slow to be a collision warning.
A spoken sentence takes well over a second to deliver, and a user walking at
1.4 m/s covers two metres in that time. So proximity rides a tone instead:

    repetition rate  ->  how close it is
    stereo pan       ->  which side it is on
    pitch            ->  obstacle, or the distinctly lower drop-off tone

Audio is generated inside PortAudio's callback thread and driven entirely by a
small block of parameters the guidance loop overwrites. Nothing here ever waits
on inference, so a 200 ms detection frame cannot stutter or delay a warning.
The callback only ever reads plain floats, which is why no lock is needed: a
torn read costs at most one slightly-wrong audio block.
"""

import math
import threading

import numpy as np

SAMPLE_RATE = 44100
_BLOCK = 512
_BEEP_MS = 60


class Sonifier:
    def __init__(self, sample_rate=SAMPLE_RATE, volume=0.35):
        self.sample_rate = sample_rate
        self.volume = volume

        self.active = False
        self.interval = 1.0
        self.frequency = 660.0
        self.pan = 0.0
        self.dropoff = False

        self._phase = 0.0
        self._since_beep = 0.0
        self._in_beep = False
        self._beep_pos = 0
        self._stream = None
        self._lock = threading.Lock()

    # -- control ------------------------------------------------------------

    def apply(self, beep):
        """Adopt a BeepState from the arbiter. Cheap enough to call per frame."""
        with self._lock:
            self.active = beep.active
            self.interval = max(beep.interval, 0.05)
            self.frequency = beep.frequency
            self.pan = max(-1.0, min(1.0, beep.pan))
            self.dropoff = beep.dropoff

    def start(self):
        import sounddevice as sd
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate, channels=2, dtype='float32',
            blocksize=_BLOCK, callback=self._callback)
        self._stream.start()
        return self

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -- synthesis ----------------------------------------------------------

    def _callback(self, outdata, frames, time_info, status):
        outdata[:] = self.render(frames)

    def render(self, frames):
        """Produce one block of stereo audio. Separated out so it is testable."""
        with self._lock:
            active, interval = self.active, self.interval
            freq, pan, dropoff = self.frequency, self.pan, self.dropoff

        block = np.zeros((frames, 2), dtype=np.float32)
        if not active:
            self._in_beep = False
            self._since_beep = 0.0
            return block

        dt = 1.0 / self.sample_rate
        beep_len = int(self.sample_rate * _BEEP_MS / 1000.0)
        mono = np.zeros(frames, dtype=np.float32)

        for i in range(frames):
            if not self._in_beep:
                self._since_beep += dt
                if self._since_beep >= interval:
                    self._in_beep = True
                    self._beep_pos = 0
                    self._since_beep = 0.0
                else:
                    continue

            # A drop-off is a two-tone chirp, so it is identifiable as a kind of
            # hazard and not merely as "something very close".
            f = freq * (1.5 if dropoff and self._beep_pos > beep_len // 2 else 1.0)
            self._phase += 2.0 * math.pi * f * dt
            # Raised-cosine envelope: a square-edged beep clicks, and clicks are
            # fatiguing to listen to for an hour.
            env = 0.5 * (1.0 - math.cos(2.0 * math.pi * self._beep_pos / beep_len))
            mono[i] = math.sin(self._phase) * env

            self._beep_pos += 1
            if self._beep_pos >= beep_len:
                self._in_beep = False

        left, right = equal_power_pan(pan)
        block[:, 0] = mono * left * self.volume
        block[:, 1] = mono * right * self.volume
        return block


def equal_power_pan(pan):
    """Constant-power stereo pan: -1 hard left, 0 centre, +1 hard right.

    Equal power rather than linear, so a hazard swinging across the front does
    not appear to dip in loudness as it passes the centre.
    """
    angle = (max(-1.0, min(1.0, pan)) + 1.0) * (math.pi / 4.0)
    return math.cos(angle), math.sin(angle)
