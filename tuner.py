#!/usr/bin/env python3
"""
Simple Acoustic Guitar Tuner (GUI + Microphone)

- Listens continuously from your default microphone
- Estimates fundamental frequency (f0) via autocorrelation + parabolic peak refinement
- Shows closest note + cents offset
- Highlights nearest standard guitar string (E2 A2 D3 G3 B3 E4)

Install deps:
  pip install numpy sounddevice

Notes:
- In macOS, you may need to grant microphone permission to your terminal/Python.
- Use your OS sound settings to pick the correct input device if needed.
"""

import math
import queue
import sys
import time
import tkinter as tk
from dataclasses import dataclass
import colorsys

import numpy as np

try:
    import sounddevice as sd
except Exception as e:
    print("Failed to import sounddevice. Install with: pip install sounddevice")
    raise

# -----------------------------
# Tuning/reference definitions
# -----------------------------

A4 = 440.0

NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Standard guitar strings (frequency in Hz, name shown)
GUITAR_STRINGS = [
    ("E2", 82.4069),
    ("A2", 110.0000),
    ("D3", 146.8324),
    ("G3", 195.9977),
    ("B3", 246.9417),
    ("E4", 329.6276),
]


@dataclass
class PitchResult:
    f0: float | None
    rms: float
    note_name: str | None
    note_freq: float | None
    cents: float | None
    string_name: str | None
    string_freq: float | None
    string_cents: float | None


def freq_to_midi(f: float) -> float:
    return 69.0 + 12.0 * math.log2(f / A4)


def midi_to_freq(m: float) -> float:
    return A4 * (2.0 ** ((m - 69.0) / 12.0))


def midi_to_note_name(m: int) -> str:
    name = NOTE_NAMES_SHARP[m % 12]
    octave = (m // 12) - 1
    return f"{name}{octave}"


def cents_off(f: float, ref: float) -> float:
    return 1200.0 * math.log2(f / ref)


# -----------------------------
# Pitch detection
# -----------------------------

def parabolic_interpolation(y: np.ndarray, x: int) -> float:
    """Refine peak position using a parabola fit around index x (y[x-1], y[x], y[x+1])."""
    if x <= 0 or x >= len(y) - 1:
        return float(x)
    y0, y1, y2 = y[x - 1], y[x], y[x + 1]
    denom = (y0 - 2.0 * y1 + y2)
    if denom == 0:
        return float(x)
    delta = 0.5 * (y0 - y2) / denom
    return float(x) + float(delta)


def estimate_f0_autocorr(
    x: np.ndarray,
    fs: float,
    fmin: float = 70.0,
    fmax: float = 400.0,
) -> float | None:
    """
    Autocorrelation-based f0 estimation.
    Works well for single notes (pluck one string).
    """
    if len(x) < 256:
        return None

    # Remove DC and apply window
    x = x.astype(np.float32)
    x = x - np.mean(x)
    window = np.hanning(len(x)).astype(np.float32)
    xw = x * window

    # Autocorrelation via FFT for speed
    n = int(2 ** np.ceil(np.log2(len(xw) * 2)))
    X = np.fft.rfft(xw, n=n)
    ac = np.fft.irfft(X * np.conj(X), n=n)
    ac = ac[: len(xw)]

    # Normalize
    ac0 = ac[0]
    if ac0 <= 1e-12:
        return None
    ac = ac / ac0

    # Only search plausible lag range
    lag_min = int(fs / fmax)
    lag_max = int(fs / fmin)
    lag_max = min(lag_max, len(ac) - 1)
    if lag_max <= lag_min + 2:
        return None

    segment = ac[lag_min:lag_max]

    # Peak pick: choose the maximum in the segment, but require it to be "peaky"
    idx = int(np.argmax(segment)) + lag_min
    peak_val = ac[idx]

    # Heuristics to avoid random noise / silence
    if peak_val < 0.15:
        return None

    # Refine lag
    lag_refined = parabolic_interpolation(ac, idx)
    if lag_refined <= 0:
        return None

    f0 = fs / lag_refined
    if not (fmin <= f0 <= fmax):
        return None
    return float(f0)


def analyze_block(x: np.ndarray, fs: float) -> PitchResult:
    # RMS for level indicator / gating
    rms = float(np.sqrt(np.mean(x.astype(np.float32) ** 2) + 1e-12))

    # Gate: don’t attempt pitch if too quiet
    if rms < 0.001:
        return PitchResult(None, rms, None, None, None, None, None, None)

    f0 = estimate_f0_autocorr(x, fs)
    if f0 is None:
        return PitchResult(None, rms, None, None, None, None, None, None)

    # Closest equal-tempered note
    midi = freq_to_midi(f0)
    midi_round = int(round(midi))
    note_freq = midi_to_freq(midi_round)
    note_name = midi_to_note_name(midi_round)
    cents = cents_off(f0, note_freq)

    # Closest guitar string target
    best = None
    for s_name, s_freq in GUITAR_STRINGS:
        sc = cents_off(f0, s_freq)
        if best is None or abs(sc) < abs(best[2]):
            best = (s_name, s_freq, sc)

    string_name, string_freq, string_cents = best
    return PitchResult(f0, rms, note_name, note_freq, cents, string_name, string_freq, string_cents)


# -----------------------------
# GUI + audio streaming
# -----------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _rgb01_to_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = rgb
    r8 = int(_clamp(r, 0.0, 1.0) * 255.0 + 0.5)
    g8 = int(_clamp(g, 0.0, 1.0) * 255.0 + 0.5)
    b8 = int(_clamp(b, 0.0, 1.0) * 255.0 + 0.5)
    return f"#{r8:02x}{g8:02x}{b8:02x}"


def _hsv_interp_black(base_rgb01: tuple[float, float, float], t: float) -> str:
    """
    Interpolate in HSV from black (off) to the base color at 100%.
    We keep H,S from the base color and scale V by t.
    """
    t = _clamp(t, 0.0, 1.0)
    h, s, v = colorsys.rgb_to_hsv(*base_rgb01)
    rgb = colorsys.hsv_to_rgb(h, s, v * t)
    return _rgb01_to_hex(rgb)


class LedHemicircle:
    """
    A semicircle ("top half of a circle") of LEDs.

    - LED index 0 is leftmost, index (n-1) is rightmost, center is at the top.
    - The lit position is driven by cents in [-50, +50] (quarter-tone flat/sharp).
    - Between two LEDs, intensity is linearly split (1-2 LEDs lit at a time).
    - LED colors: center green, adjacent two yellow, rest red; off is black.
    - Intensity interpolation is done in HSV from black to 100%.
    """

    def __init__(self, parent: tk.Widget, led_count: int = 11):
        if led_count < 3 or led_count % 2 == 0:
            raise ValueError("led_count must be an odd integer >= 3")

        self.led_count = led_count
        self.center_idx = led_count // 2
        self.max_cents = 50.0  # quarter-tone

        self.canvas = tk.Canvas(parent, height=210, highlightthickness=0)
        self._led_items: list[int] = []
        self._text_item: int | None = None
        self._note: str | None = None
        self._cents: float | None = None

        self.canvas.bind("<Configure>", self._on_resize)
        self._rebuild()

    def pack(self, **kwargs):
        self.canvas.pack(**kwargs)

    def set_value(self, note: str | None, cents: float | None):
        self._note = note
        self._cents = cents
        self._update_led_fills()
        self._update_text()

    def reset(self):
        self.set_value(None, None)

    def _base_color_rgb01(self, idx: int) -> tuple[float, float, float]:
        d = abs(idx - self.center_idx)
        if d == 0:
            return (0.0, 1.0, 0.0)  # green
        if d == 1:
            return (1.0, 1.0, 0.0)  # yellow
        return (1.0, 0.0, 0.0)  # red

    def _cents_to_led_intensities(self, cents: float | None) -> list[float]:
        intensities = [0.0] * self.led_count
        if cents is None:
            return intensities

        c = _clamp(float(cents), -self.max_cents, self.max_cents)
        side = self.center_idx
        pos = self.center_idx + (c / self.max_cents) * side
        pos = _clamp(pos, 0.0, float(self.led_count - 1))

        i0 = int(math.floor(pos))
        frac = float(pos - i0)
        i1 = min(i0 + 1, self.led_count - 1)

        if i0 == i1:
            intensities[i0] = 1.0
        else:
            intensities[i0] = 1.0 - frac
            intensities[i1] = frac
        return intensities

    def _update_led_fills(self):
        intensities = self._cents_to_led_intensities(self._cents)
        for idx, item in enumerate(self._led_items):
            base = self._base_color_rgb01(idx)
            fill = _hsv_interp_black(base, intensities[idx])
            self.canvas.itemconfigure(item, fill=fill, outline=fill)

    def _update_text(self):
        if self._text_item is None:
            return

        if self._note is None or self._cents is None:
            txt = ""
        else:
            c = float(self._cents)
            txt = f"{self._note}\n{c:+.1f}"

        self.canvas.itemconfigure(self._text_item, text=txt)

    def _rebuild(self):
        self.canvas.delete("all")
        self._led_items.clear()

        # Geometry
        w = max(int(self.canvas.winfo_width()), 1)
        h = max(int(self.canvas.winfo_height()), 1)
        margin = 18
        cx = w / 2.0
        cy = h - margin
        radius = max(30.0, min((w / 2.0) - margin, h - (margin * 2.5)))
        led_r = _clamp(radius * 0.075, 7.0, 14.0)

        # LEDs along the top semicircle (left -> right)
        for i in range(self.led_count):
            angle_deg = 180.0 - (180.0 * i / (self.led_count - 1))
            a = math.radians(angle_deg)
            x = cx + radius * math.cos(a)
            y = cy - radius * math.sin(a)
            item = self.canvas.create_oval(
                x - led_r,
                y - led_r,
                x + led_r,
                y + led_r,
                fill="#000000",
                outline="#000000",
            )
            self._led_items.append(item)

        # Center text
        text_y = cy - radius * 0.58
        self._text_item = self.canvas.create_text(
            cx,
            text_y,
            text="",
            fill="#ffffff",
            font=("Helvetica", 20, "bold"),
            justify="center",
        )

        self._update_led_fills()
        self._update_text()

    def _on_resize(self, _evt=None):
        self._rebuild()


class GuitarTunerApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Guitar Tuner")

        self.fs = 44100
        self.block_seconds = 0.20  # analysis window
        self.block_size = int(self.fs * self.block_seconds)

        self.q: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
        self.stream = None
        self.last_update = 0.0
        self._stopping = False
        self._auto_started = False
        self._frame_loop_running = False

        # Smoothed display state (for 60 FPS rendering)
        self._prev_analyzed_cents: float | None = None  # for MA2
        self._interp_note: str | None = None
        self._interp_from_cents: float | None = None
        self._interp_to_cents: float | None = None
        self._interp_t0: float = 0.0
        self._display_cents: float | None = None

        # UI
        self.note_var = tk.StringVar(value="")
        self.freq_var = tk.StringVar(value="Freq: — Hz")
        self.cents_var = tk.StringVar(value="Cents: —")
        self.string_var = tk.StringVar(value="String: —")
        self.level_var = tk.StringVar(value="Level: —")

        big = ("Helvetica", 44, "bold")
        mid = ("Helvetica", 16)
        small = ("Helvetica", 12)

        # self.note_label = tk.Label(self.root, textvariable=self.note_var, font=big)
        # self.note_label.pack(padx=16, pady=(14, 4))
        self.note_label = None

        # self.string_label = tk.Label(self.root, textvariable=self.string_var, font=mid)
        # self.string_label.pack(padx=16, pady=(0, 10))
        #
        # self.freq_label = tk.Label(self.root, textvariable=self.freq_var, font=mid)
        # self.freq_label.pack(padx=16, pady=(0, 2))
        #
        # self.cents_label = tk.Label(self.root, textvariable=self.cents_var, font=mid)
        # self.cents_label.pack(padx=16, pady=(0, 10))

        # LED tuner visualization (bottom section)
        self.led_arc = LedHemicircle(self.root, led_count=11)
        self.led_arc.pack(padx=12, pady=(0, 10), fill="x")

        # self.level_label = tk.Label(self.root, textvariable=self.level_var, font=small)
        # self.level_label.pack(padx=16, pady=(0, 14))

        self.status = tk.StringVar(value="Stopped")
        # self.status_label = tk.Label(self.root, textvariable=self.status, font=small)
        # self.status_label.pack(padx=16, pady=(0, 12))
        self.status_label = None

        # Color behavior
        self.ok_bg = self.root.cget("bg")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Map>", self._on_first_map)

    def _on_first_map(self, _evt=None):
        if self._auto_started:
            return
        self._auto_started = True
        # Once the window is visible, start the stream.
        self.root.after(50, self.start)

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            # Drop on overload rather than blocking
            pass
        x = indata[:, 0].copy()  # mono
        try:
            self.q.put_nowait(x)
        except queue.Full:
            pass

    def start(self):
        if self.stream is not None or self._stopping:
            return

        self.status.set("Starting microphone…")
        # Defer actual stream start to the next Tk tick (can block briefly on macOS).
        self.root.after(1, self._start_stream_impl)

    def _start_stream_impl(self):
        if self.stream is not None or self._stopping:
            return

        # Clear any stale audio blocks.
        with self.q.mutex:
            self.q.queue.clear()

        # Reset smoothing state on each start.
        self._prev_analyzed_cents = None
        self._interp_note = None
        self._interp_from_cents = None
        self._interp_to_cents = None
        self._display_cents = None
        self._interp_t0 = time.perf_counter()

        try:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self.fs,
                blocksize=self.block_size,
                dtype="float32",
                callback=self.audio_callback,
            )
            self.stream.start()
        except Exception as e:
            self.status.set(f"Failed to start mic: {e}")
            self.stream = None
            return

        self.status.set("Listening… Pluck one string at a time.")
        self.root.after(10, self.update_loop)
        if not self._frame_loop_running:
            self._frame_loop_running = True
            self.root.after(0, self.frame_loop)

    def stop(self):
        if self.stream is None or self._stopping:
            return

        # Stop synchronously (no buttons to update; this is also used by window-close).
        self._stopping = True
        self.status.set("Stopping…")
        self._stop_stream_impl()

    def _stop_stream_impl(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        self._stopping = False

        self.status.set("Stopped")
        self.note_var.set("")
        self.freq_var.set("Freq: — Hz")
        self.cents_var.set("Cents: —")
        self.string_var.set("String: —")
        self.level_var.set("Level: —")
        if self.note_label is not None:
            self.note_label.config(bg=self.ok_bg)
        self.led_arc.reset()

    def frame_loop(self):
        """
        Render at ~60 FPS using interpolation toward the latest analyzed (MA2) cents value.
        Analysis still runs at the audio callback block rate (block_seconds).
        """
        if self.stream is None:
            self._frame_loop_running = False
            return

        if self._interp_note is None or self._interp_to_cents is None or self._interp_from_cents is None:
            # No tone detected yet (or currently silent).
            self.led_arc.reset()
        else:
            now = time.perf_counter()
            u = (now - self._interp_t0) / max(self.block_seconds, 1e-6)
            u = _clamp(u, 0.0, 1.0)
            c = float(self._interp_from_cents) + (float(self._interp_to_cents) - float(self._interp_from_cents)) * float(u)
            self._display_cents = c
            self.led_arc.set_value(self._interp_note, c)

        self.root.after(16, self.frame_loop)

    def update_loop(self):
        if self.stream is None:
            return

        # Drain queue, analyze the newest block (lower latency, steadier display)
        x = None
        while True:
            try:
                x = self.q.get_nowait()
            except queue.Empty:
                break

        if x is not None:
            res = analyze_block(x, self.fs)
            self.render(res)

        self.root.after(30, self.update_loop)

    def render(self, res: PitchResult):
        # Level bar-ish
        level_db = 20.0 * math.log10(max(res.rms, 1e-6))
        self.level_var.set(f"Level: {level_db:5.1f} dBFS")

        if res.f0 is None:
            self.note_var.set("")
            self.freq_var.set("Freq: — Hz")
            self.cents_var.set("Cents: —")
            self.string_var.set("String: —")
            if self.note_label is not None:
                self.note_label.config(bg=self.ok_bg)
            self.led_arc.reset()

            # Reset interpolation state when no pitch is detected.
            self._prev_analyzed_cents = None
            self._interp_note = None
            self._interp_from_cents = None
            self._interp_to_cents = None
            self._display_cents = None
            return

        self.note_var.set(res.note_name or "—")
        self.freq_var.set(f"Freq: {res.f0:7.2f} Hz")

        # Prefer cents vs string target (more useful for tuning)
        sc = res.string_cents if res.string_cents is not None else res.cents
        direction = "flat" if sc is not None and sc < -1.0 else ("sharp" if sc is not None and sc > 1.0 else "in tune")
        if sc is None:
            self.cents_var.set("Cents: —")
        else:
            self.cents_var.set(f"Cents vs {res.string_name}: {sc:+6.1f} ({direction})")

        self.string_var.set(f"Nearest string: {res.string_name} ({res.string_freq:.1f} Hz)")

        # LED arc visualization uses cents vs closest equal-tempered note.
        # Apply a 2-sample moving average (MA2) prefilter.
        if res.cents is None:
            self._prev_analyzed_cents = None
            self._interp_note = None
            self._interp_from_cents = None
            self._interp_to_cents = None
            self._display_cents = None
        else:
            current_cents = float(res.cents)
            if self._prev_analyzed_cents is None:
                ma2 = current_cents
            else:
                ma2 = 0.5 * (current_cents + float(self._prev_analyzed_cents))
            self._prev_analyzed_cents = current_cents

            # Interpolate from current displayed value to the new MA2 target over block_seconds.
            if self._display_cents is None:
                self._display_cents = ma2
            self._interp_note = res.note_name
            self._interp_from_cents = float(self._display_cents)
            self._interp_to_cents = float(ma2)
            self._interp_t0 = time.perf_counter()

        # Simple “traffic light”
        if sc is None:
            if self.note_label is not None:
                self.note_label.config(bg=self.ok_bg)
        else:
            if abs(sc) <= 5.0:
                if self.note_label is not None:
                    self.note_label.config(bg="#b6f2b6")  # light green
            elif abs(sc) <= 15.0:
                if self.note_label is not None:
                    self.note_label.config(bg="#f7f0b2")  # light yellow
            else:
                if self.note_label is not None:
                    self.note_label.config(bg="#f6b3b3")  # light red

    def on_close(self):
        # Ensure the stream is stopped before tearing down Tk.
        if self.stream is not None:
            self.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    app = GuitarTunerApp()
    app.run()


if __name__ == "__main__":
    main()
