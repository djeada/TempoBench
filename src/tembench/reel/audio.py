"""A soundtrack for a reel, synthesised from the same story and timeline.

Nothing is sampled or licensed: every sound is built here, and every event is
placed on the frame it belongs to.

- a warm pad in D major carries the reel, moving from rest (measure) through
  tension (fit) to resolution (verdict);
- each trial lands with a soft marimba pluck, pitched by input size, so a
  sweep is heard climbing; a size's median lands with a lower bell;
- each class swept in by the fit is a breath of filtered noise, then a note:
  dull and low for a class that misses badly, bright for one that fits;
- the winner gets a rising arpeggio; the verdict blooms into the home chord;
- the collapse is a falling glide that lands on a single unison note — the
  lines becoming one, heard;
- everything shares one small room (a convolution reverb) and a soft limiter.
"""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .render import COLLAPSE_BACK, COLLAPSE_FALL, COLLAPSE_HOLD, COLLAPSE_START

if TYPE_CHECKING:
    from .render import Reel

RATE = 44100
#: D major pentatonic (D E F♯ A B), as semitones above D.
PENTATONIC = (0, 2, 4, 7, 9)
D4 = 62

#: The pad's chords, as MIDI notes: home, relative minor, subdominant, and an
#: unresolved suspended dominant for while the classes are being tried.
D_MAJ9 = (50, 57, 62, 66, 69, 76)
B_MIN9 = (47, 54, 62, 66, 69, 73)
G_MAJ9 = (43, 55, 62, 66, 69, 71)
A_SUS = (45, 57, 62, 64, 69, 74)


def hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def scale_note(step: int, root: int = D4) -> int:
    """The `step`-th note of D major pentatonic upwards from `root`."""
    octave, k = divmod(step, len(PENTATONIC))
    return root + 12 * octave + PENTATONIC[k]


class _Mix:
    """A stereo buffer that sounds are added into at given times."""

    def __init__(self, seconds: float):
        self.length = int(math.ceil(seconds * RATE)) + RATE // 2
        self.dry = np.zeros((self.length, 2))
        self.pad = np.zeros((self.length, 2))

    def add(self, at: float, mono: np.ndarray, pan: float = 0.0, bus: str = "dry", gain: float = 1.0) -> None:
        """Mix `mono` in from `at` seconds, panned -1 (left) … 1 (right) at equal power."""
        start = int(at * RATE)
        if start >= self.length or start + len(mono) <= 0:
            return
        lo = max(0, -start)
        end = min(self.length, start + len(mono))
        chunk = mono[lo: lo + end - max(start, 0)] * gain
        angle = (pan + 1) * math.pi / 4
        target = self.dry if bus == "dry" else self.pad
        target[max(start, 0): end, 0] += chunk * math.cos(angle)
        target[max(start, 0): end, 1] += chunk * math.sin(angle)


def _t(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * RATE)) / RATE


def pluck(freq: float, decay: float = 0.45) -> np.ndarray:
    """A soft mallet: a pure tone with a woody, quickly fading overtone."""
    t = _t(decay * 6)
    attack = np.minimum(1.0, t / 0.004)
    body = np.sin(2 * np.pi * freq * t) * np.exp(-t / decay)
    wood = 0.28 * np.sin(2 * np.pi * freq * 4.0 * t) * np.exp(-t / (decay * 0.12))
    shimmer = 0.2 * np.sin(2 * np.pi * freq * 2.0 * t) * np.exp(-t / (decay * 0.5))
    return attack * (body + wood + shimmer)


def bell(freq: float, decay: float = 1.4) -> np.ndarray:
    """A glassy bell: inharmonic partials that ring at different lengths."""
    t = _t(decay * 5)
    partials = ((1.0, 1.0, 1.0), (2.76, 0.35, 0.45), (5.4, 0.15, 0.25), (2.0, 0.25, 0.7))
    tone = sum(a * np.sin(2 * np.pi * freq * r * t) * np.exp(-t / (decay * d)) for r, a, d in partials)
    return np.minimum(1.0, t / 0.003) * tone / 1.75


def breath(seconds: float, rng: np.random.Generator, brightness: float = 0.5) -> np.ndarray:
    """A swell of soft, low-passed noise — the sound of a curve sweeping past."""
    n = int(seconds * RATE)
    noise = rng.standard_normal(n)
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n, 1 / RATE)
    cutoff = 400 + 2600 * brightness
    spectrum *= 1 / (1 + (freqs / cutoff) ** 4) * (freqs > 120)
    shaped = np.fft.irfft(spectrum, n)
    shaped /= np.max(np.abs(shaped)) or 1.0
    envelope = np.sin(np.pi * np.linspace(0, 1, n)) ** 2
    return shaped * envelope


def glide(start_hz: float, end_hz: float, seconds: float) -> np.ndarray:
    """A sine that slides between two pitches, easing like the lines it follows."""
    t = _t(seconds)
    p = t / seconds
    eased = p * p * (3 - 2 * p)
    freq = start_hz * (end_hz / start_hz) ** eased
    phase = 2 * np.pi * np.cumsum(freq) / RATE
    envelope = np.minimum(1.0, t / 0.08) * (1 - 0.4 * p)
    return np.sin(phase) * envelope + 0.2 * np.sin(2 * phase) * envelope


def _pad_chord(mix: _Mix, notes: tuple[int, ...], start: float, end: float, gain: float) -> None:
    """Hold a chord from `start` to `end`, with slow swells at both ends."""
    length = end - start + 1.6
    t = _t(length)
    swell = np.minimum(1.0, t / 1.1) * np.clip((length - t) / 1.4, 0, 1)
    for k, midi in enumerate(notes):
        # Low voices are felt more than heard on a phone, and muddy earbuds.
        level = gain * (1.0 if midi > 52 else 0.55) / len(notes) ** 0.5
        for cents, pan in ((-6, -0.55), (0, 0.0), (6, 0.55)):
            f = hz(midi) * 2 ** (cents / 1200)
            phase = 2 * np.pi * f * t + k
            tone = np.sin(phase) + 0.22 * np.sin(2 * phase) + 0.06 * np.sin(3 * phase)
            # A slow tremolo per voice keeps the pad breathing.
            tremolo = 1 + 0.08 * np.sin(2 * np.pi * (0.13 + 0.05 * k) * t + cents)
            mix.add(start - 0.4, tone * swell * tremolo, pan, bus="pad", gain=level / 3)


def _lowpass(signal: np.ndarray, cutoff: float) -> np.ndarray:
    spectrum = np.fft.rfft(signal, axis=0)
    freqs = np.fft.rfftfreq(signal.shape[0], 1 / RATE)
    spectrum *= (1 / (1 + (freqs / cutoff) ** 2))[:, None]
    return np.fft.irfft(spectrum, signal.shape[0], axis=0)


def _air(signal: np.ndarray, corner: float = 3000.0, lift: float = 0.8) -> np.ndarray:
    """A gentle high shelf: sparkle on the mallets and bells, no harshness."""
    spectrum = np.fft.rfft(signal, axis=0)
    freqs = np.fft.rfftfreq(signal.shape[0], 1 / RATE)
    ratio = (freqs / corner) ** 2
    spectrum *= (1 + lift * ratio / (1 + ratio))[:, None]
    return np.fft.irfft(spectrum, signal.shape[0], axis=0)


def _highpass(signal: np.ndarray, cutoff: float) -> np.ndarray:
    """Remove rumble a phone speaker cannot play and earbuds turn to mud."""
    spectrum = np.fft.rfft(signal, axis=0)
    freqs = np.fft.rfftfreq(signal.shape[0], 1 / RATE)
    ratio = (freqs / cutoff) ** 4
    spectrum *= (ratio / (1 + ratio))[:, None]
    return np.fft.irfft(spectrum, signal.shape[0], axis=0)


def _reverb(signal: np.ndarray, rng: np.random.Generator, seconds: float = 2.4, wet: float = 0.24) -> np.ndarray:
    """A small, warm room: convolution with decaying, darkened noise per channel."""
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    impulse = rng.standard_normal((n, 2)) * np.exp(-t / 0.55)[:, None]
    impulse = _lowpass(impulse, 3500.0)
    impulse[: int(0.012 * RATE)] = 0  # pre-delay keeps the dry hit crisp
    impulse /= np.sqrt(np.sum(impulse**2, axis=0))
    size = signal.shape[0] + n
    fft_n = 1 << (size - 1).bit_length()
    out = np.fft.irfft(np.fft.rfft(signal, fft_n, axis=0) * np.fft.rfft(impulse, fft_n, axis=0), fft_n, axis=0)
    return signal + wet * out[: signal.shape[0]]


def soundtrack(reel: Reel, seed: int = 7) -> np.ndarray:
    """The reel's soundtrack: float stereo samples at `RATE`, peaking just under 1."""
    rng = np.random.default_rng(seed)
    story, timeline = reel.story, reel.timeline
    pace = timeline.pace
    start = dict(zip([name for name, _ in reel.acts], reel.starts))
    end = reel.duration
    mix = _Mix(end)

    # The pad: rest, a step away, tension while the classes compete, home.
    measure_mid = start["measure"] + timeline.measure / 2
    fit_mid = start["fit"] + (start["verdict"] - start["fit"]) / 2
    for notes, a, b in (
        (D_MAJ9, 0.0, measure_mid),
        (B_MIN9, measure_mid, start["fit"]),
        (G_MAJ9, start["fit"], fit_mid),
        (A_SUS, fit_mid, start["verdict"]),
        (D_MAJ9, start["verdict"], end),
    ):
        _pad_chord(mix, notes, a, b, gain=0.14)

    # Hook: two bell notes under the fact and the question.
    mix.add(start["hook"] + 0.1 * pace, bell(hz(scale_note(5))), 0.0, gain=0.22)
    mix.add(start["hook"] + 0.9 * pace, bell(hz(scale_note(7))), 0.0, gain=0.22)

    # Measure: one pluck per trial, pitched by input size.
    sizes = sorted({x for s in story.series for x, _ in s.points})
    rank = {x: k for k, x in enumerate(sizes)}
    pans = np.linspace(-0.6, 0.6, len(story.series)) if len(story.series) > 1 else [0.0]
    last = -1.0
    for k, trial in enumerate(story.trials):
        at = start["measure"] + reel.trial_at[k]
        gap = at - last
        if gap < 0.045:  # a flood of runs is a texture, not a hail of clicks
            continue
        note = scale_note(3 + rank[trial.x] * 9 // max(1, len(sizes) - 1) if len(sizes) > 1 else 5)
        mix.add(at, pluck(hz(note), 0.32), float(pans[trial.series]), gain=0.17 * min(1.0, (gap / 0.12) ** 0.5))
        last = at
    for (series, x), landed in reel.median_at.items():
        note = scale_note(rank[x] * 9 // max(1, len(sizes) - 1) if len(sizes) > 1 else 3) - 12
        mix.add(start["measure"] + landed + 0.05, bell(hz(note), 0.9), float(pans[series]), gain=0.09)

    # Fit: a breath as each class sweeps in, then its verdict in one note.
    per = timeline.per_class
    for j, model in enumerate(reel.models):
        at = start["fit"] + j * per
        errors = [s.candidates_by_model[model].error for s in story.series if model in s.candidates_by_model]
        error = sum(errors) / len(errors) if errors else 1.0
        good = 1 - math.sqrt(min(error, 1.0))
        mix.add(at, breath(per * 0.5, rng, brightness=good), 0.0, gain=0.10)
        note = scale_note(int(round(2 + good * 9)))
        mix.add(at + per * 0.38, pluck(hz(note), 0.25 + 0.5 * good), (j % 2) * 0.4 - 0.2, gain=0.12 + 0.12 * good)
    decided = start["fit"] + per * len(reel.models)
    for k, step in enumerate((5, 7, 8, 10, 12)):
        mix.add(decided + k * 0.075 * pace, pluck(hz(scale_note(step)), 0.5), -0.4 + 0.2 * k, gain=0.16)
    mix.add(decided + 0.4 * pace, bell(hz(scale_note(15)), 1.6), 0.0, gain=0.10)

    # Verdict: the home chord blooms as the class appears.
    v = start["verdict"]
    mix.add(v, bell(hz(50), 2.5), 0.0, gain=0.22)  # a low D underneath
    for k, midi in enumerate((74, 78, 81, 85)):
        mix.add(v + (0.9 + 0.06 * k) * pace, bell(hz(midi), 1.8), -0.3 + 0.2 * k, gain=0.11)
    if reel.factors is not None:
        fall_at = v + COLLAPSE_START * pace
        mix.add(fall_at, glide(hz(81), hz(62), COLLAPSE_FALL * pace), 0.0, gain=0.10)
        landed = fall_at + COLLAPSE_FALL * pace
        mix.add(landed, bell(hz(74), 2.2), -0.5, gain=0.16)  # every voice on one note:
        mix.add(landed, bell(hz(74), 2.2), 0.5, gain=0.16)  # the lines are one
        mix.add(landed, bell(hz(50), 1.5), 0.0, gain=0.18)
        back = landed + COLLAPSE_HOLD * pace
        mix.add(back, glide(hz(69), hz(81), COLLAPSE_BACK * pace), 0.0, gain=0.06)
    cards_at = v + (COLLAPSE_START + COLLAPSE_FALL + COLLAPSE_HOLD + COLLAPSE_BACK if reel.factors else 1.2) * pace
    for k in range(len(reel.cards)):
        mix.add(cards_at + 0.3 * k * pace, pluck(hz(scale_note(10 + 2 * k)), 0.4), 0.0, gain=0.09)

    # Outro: one last bell, left to ring out.
    mix.add(start["outro"], bell(hz(scale_note(10)), 2.0), 0.0, gain=0.16)

    out = _lowpass(mix.pad, 1800.0) + _air(mix.dry)
    out = _highpass(_reverb(out, rng), 70.0)
    t = np.arange(out.shape[0]) / RATE
    fade = np.minimum(1.0, t / 0.2) * np.clip((end + 0.3 - t) / 1.0, 0, 1)
    out *= fade[:, None]
    out = np.tanh(1.2 * out / max(1e-9, float(np.max(np.abs(out)))))
    out *= 0.84 / max(1e-9, float(np.max(np.abs(out))))  # -1.5 dBFS: headroom for AAC
    return out[: int(end * RATE)]


def write_wav(samples: np.ndarray, path: Path) -> None:
    """16-bit stereo PCM."""
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), mode="wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(pcm.tobytes())
