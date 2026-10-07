"""Drawing a `Story` as a vertical video, one matplotlib frame at a time.

The reel is five acts:

hook     the surprising fact, posed as a question ("python is 18× slower
         than cpp. Does it scale worse?");
measure  the run replayed: every trial drops into place in the order it ran;
fit      one curve per series morphs through every complexity class, tied to
         the medians by lines that show how far each class misses;
verdict  the bound and the class, then, when the series share a class, each
         one divided by its constant factor until they lie on one curve —
         what "the same Big-O" means — before the result cards;
outro    a closing line.

Every frame is a pure function of time: `Reel.draw(t)` puts each artist where
it belongs at `t` seconds, so any single frame (the poster, a test) can be
drawn without playing the ones before it.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
import textwrap
from bisect import bisect_right
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Callable

from matplotlib import rcParams
from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullLocator

from .story import Series, Story, pretty_model

BG = "#0e0e0d"
SURFACE = "#171716"
TEXT = "#f2f1ec"
MUTED = "#a3a299"
FAINT = "#5c5b55"
GRID = "#262624"
ACCENT = "#5b9cec"
GOOD = "#5cc68a"
CONFIDENCE_COLOR = {"high": GOOD, "medium": "#e0a43a", "low": "#f07a72"}

#: Each caveat in a few words, short enough for a card.
CAVEAT_SHORT = {
    "few-points": "few input sizes",
    "narrow-n-range": "narrow size range",
    "flat-signal": "times barely grew",
    "wide-exponent-ci": "uncertain exponent",
    "overhead-dominated": "mostly startup cost",
    "ambiguous-class": "close runner-up",
    "poor-fit": "no class fits well",
    "constant-class": "flat curve",
    "exponent-mismatch": "between two classes",
    "single-point-growth": "growth in one reading",
    "outlier-dropped": "outlier dropped",
    "non-positive-timings": "below timer resolution",
    "thin-samples": "few runs per size",
    "unstable-timings": "noisy runs",
}

FONT = ["Inter", "DejaVu Sans"]
MONO = ["DejaVu Sans Mono"]

#: Figure size in inches; the pixel size comes from the dpi.  9:16 is the
#: shape of a phone screen held upright.
FIG_W, FIG_H = 9.0, 16.0
#: Where the chart sits on the figure: left, bottom, width, height.
CHART = (0.15, 0.445, 0.78, 0.36)

#: The collapse beat inside the verdict, in seconds at normal speed: when the
#: lines start to fall, how long the fall takes, how long they stay together,
#: and how long they take to spring back.
COLLAPSE_START, COLLAPSE_FALL, COLLAPSE_HOLD, COLLAPSE_BACK = 1.8, 1.0, 1.2, 0.6


@dataclass(frozen=True)
class Timeline:
    """How long each act lasts, in seconds."""

    hook: float = 2.6
    measure: float = 6.5
    per_class: float = 1.0
    verdict: float = 4.5
    #: Added to the verdict when the series share a class and collapse.
    collapse: float = 3.2
    outro: float = 1.6
    #: Multiplies every beat inside the acts, so a faster reel stays in step.
    pace: float = 1.0

    def faster(self, speed: float) -> Timeline:
        """The same reel played `speed` times as fast."""
        return Timeline(*(getattr(self, f.name) / speed for f in fields(self)))

    def acts(self, classes: int, collapses: bool) -> list[tuple[str, float]]:
        return [
            ("hook", self.hook),
            ("measure", self.measure),
            ("fit", self.per_class * (classes + 1)),
            ("verdict", self.verdict + (self.collapse if collapses else 0.0)),
            ("outro", self.outro),
        ]


def _ease(p: float) -> float:
    """Ease out (cubic): fast start, gentle landing."""
    p = min(1.0, max(0.0, p))
    return 1 - (1 - p) ** 3


def _smooth(p: float) -> float:
    """Ease in and out."""
    p = min(1.0, max(0.0, p))
    return p * p * (3 - 2 * p)


def human(value: float) -> str:
    """Axis numbers a viewer reads at a glance: 1k, 20k, 1M, 0.5, 120."""
    if value == 0:
        return "0"
    for limit, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(value) >= limit:
            return f"{value / limit:g}{suffix}"
    return f"{value:g}"


def _ms(value: float) -> str:
    if value >= 1000:
        return f"{value / 1000:.2f} s"
    if value >= 100:
        return f"{value:.0f} ms"
    if value >= 1:
        return f"{value:.1f} ms"
    if value >= 0.01:
        return f"{value:.3f} ms"
    return f"{value * 1000:.1f} µs"


def _duration(seconds: float) -> str:
    if seconds >= 60:
        return f"{int(seconds // 60)} min {int(seconds % 60)} s"
    return f"{seconds:.1f} s" if seconds < 10 else f"{seconds:.0f} s"


def _times(ratio: float) -> str:
    if ratio >= 100:
        return f"{ratio:,.0f}×"
    return f"{ratio:.0f}×" if ratio >= 10 else f"{ratio:.1f}×"


def _blend(a: float, b: float, p: float, log: bool) -> float:
    """Interpolate between two values — geometrically on a log axis."""
    if log and a > 0 and b > 0:
        return math.exp((1 - p) * math.log(a) + p * math.log(b))
    return (1 - p) * a + p * b


def _percent(error: float) -> str:
    return "<1%" if error < 0.01 else f"{error:.0%}"


class Reel:
    """The figure, its artists, and how they move."""

    def __init__(self, story: Story, timeline: Timeline | None = None, dpi: float = 120.0):
        self.story = story
        self.timeline = timeline or Timeline()
        self.models = story.models
        self.factors = story.constant_factors()
        self.gap = story.speed_gap()
        if self.gap and self.factors:
            # One number for one claim: the hook, the headline and the
            # collapse all quote the constant factor, not the gap at one size.
            slow, fast, _ = self.gap
            index = {id(s): i for i, s in enumerate(story.series)}
            self.gap = (slow, fast, self.factors[index[id(slow)]] / self.factors[index[id(fast)]])
        self.acts = self.timeline.acts(len(self.models), self.factors is not None)
        self.starts = [0.0]
        for _, length in self.acts:
            self.starts.append(self.starts[-1] + length)
        self.duration = self.starts[-1]
        self.collapse = 0.0

        self.fig = Figure(figsize=(FIG_W, FIG_H), dpi=dpi, facecolor=BG)
        self._build_header()
        self._build_chart()
        self._build_panel()
        self._schedule_trials()

    def _fade(self, local: float, start: float, length: float = 0.4) -> float:
        """0 → 1 over `length` seconds from `start`, both at normal speed."""
        pace = self.timeline.pace
        return _ease((local - start * pace) / (length * pace))

    # ---- layout ---------------------------------------------------------

    def _text(self, x: float, y: float, s: str, **kw) -> Text:
        kw.setdefault("color", TEXT)
        kw.setdefault("family", FONT)
        return self.fig.text(x, y, s, **kw)

    def _build_header(self) -> None:
        story = self.story
        self._text(0.07, 0.957, "TEMPOBENCH", size=13, weight="bold", color=ACCENT)
        self._text(0.07, 0.915, story.title, size=34, weight="bold", va="baseline")
        # A legend that never leaves the screen: the series are the cast.
        self.legend: list[Text] = []
        x = 0.07
        for s in story.series:
            if not s.name:
                continue
            self.legend.append(self._text(x, 0.884, "●", size=16, color=s.color, va="baseline"))
            self.legend.append(self._text(x + 0.03, 0.884, s.name, size=17, color=MUTED, va="baseline"))
            x += 0.08 + 0.0205 * len(s.name)
        self.steps = [
            self._text(0.07 + 0.29 * i, 0.840, name, size=15, weight="bold", color=FAINT)
            for i, name in enumerate(("Measure", "Fit", "Verdict"))
        ]
        self.step_fill: list[Rectangle] = []
        for i in range(3):
            for colour, store in ((GRID, None), (ACCENT, self.step_fill)):
                bar = Rectangle((0.07 + 0.29 * i, 0.829), 0.26, 0.0028, transform=self.fig.transFigure, color=colour)
                self.fig.add_artist(bar)
                if store is not None:
                    store.append(bar)

        middle = CHART[1] + CHART[3] / 2
        self.hook_fact = self._text(0.5, middle + 0.055, "", size=36, weight="bold", ha="center", va="center",
                                    linespacing=1.25)
        self.hook_question = self._text(0.5, middle - 0.07, "", size=30, weight="bold", ha="center", va="center",
                                        color=ACCENT)
        fact, question = self._hook_lines()
        self.hook_fact.set_text(fact)
        self.hook_question.set_text(question)
        self.caption = self._text(0.07, 0.392, "", size=18, va="top", linespacing=1.45)
        self.footer = self._text(0.5, 0.025, "", size=12, color=FAINT, ha="center")

    def _hook_lines(self) -> tuple[str, str]:
        story = self.story
        if self.gap and self.gap[2] >= 1.5:
            slow, fast, ratio = self.gap
            return textwrap.fill(f"{slow.name} is {_times(ratio)} slower than {fast.name}.", 22), "Does it scale worse?"
        return textwrap.fill(story.title, 22), f"What happens as {story.x_name} grows?"

    def _build_chart(self) -> None:
        story = self.story
        ax = self.ax = self.fig.add_axes(CHART, facecolor=SURFACE)
        xs = [x for s in story.series for x, _ in s.points]
        ys = [y for s in story.series for _, y in s.points] + [t.y for t in story.trials]
        if story.log_x:
            ax.set_xscale("log")
            ax.set_xlim(min(xs) / 1.35, max(xs) * 1.35)
        else:
            pad = (max(xs) - min(xs)) * 0.06 or 1.0
            ax.set_xlim(min(xs) - pad, max(xs) + pad)
        if story.log_y:
            ax.set_yscale("log")
            ax.set_ylim(min(ys) / 2.5, max(ys) * 2.5)
        else:
            ax.set_ylim(0, max(ys) * 1.2)
        for axis, log in ((ax.xaxis, story.log_x), (ax.yaxis, story.log_y)):
            lo, hi = axis.get_view_interval()
            decades = math.log10(hi / lo) if log else 0
            axis.set_major_locator(
                LogLocator(10, subs=(1.0, 2.0, 5.0) if decades < 2.5 else (1.0,)) if log else MaxNLocator(5)
            )
            axis.set_minor_locator(NullLocator())
            axis.set_major_formatter(FuncFormatter(lambda v, _: human(v)))
        ax.tick_params(colors=MUTED, labelsize=14, length=0, pad=8)
        for side, spine in ax.spines.items():
            spine.set_visible(side in ("left", "bottom"))
            spine.set_color(FAINT)
        ax.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.set_xlabel(f"Input size ({story.x_name})", color=MUTED, size=15, family=FONT, labelpad=10)
        ax.set_ylabel(story.y_label, color=MUTED, size=15, family=FONT, labelpad=10)

        self.trial_dots = [ax.scatter([], [], color=s.color, alpha=0.55, linewidths=0, zorder=3) for s in story.series]
        self.medians = [ax.scatter([], [], color=s.color, edgecolors=BG, linewidths=2, zorder=6) for s in story.series]
        # While a class is tried, each median is tied to its curve: how far
        # the class misses is something to see, not just a number.
        self.misses = [LineCollection([], colors=s.color, linewidths=2.2, alpha=0.75, zorder=4) for s in story.series]
        for collection in self.misses:
            ax.add_collection(collection)
        self.fit_lines = [ax.plot([], [], color=s.color, linewidth=3.2, zorder=5, solid_capstyle="round")[0]
                          for s in story.series]
        self.glows = [ax.plot([], [], color=s.color, linewidth=16, alpha=0.0, zorder=4)[0] for s in story.series]
        self.bounds = [ax.plot([], [], color=s.color, linewidth=4.5, zorder=5, solid_capstyle="round")[0]
                       for s in story.series]

        # The class on trial (and then the verdict), large in the corner.
        self.class_label = ax.text(0.04, 0.95, "", transform=ax.transAxes, size=34, weight="bold", family=FONT,
                                   color=TEXT, va="top", zorder=9)
        self.class_sub = ax.text(0.04, 0.80, "", transform=ax.transAxes, size=17, family=FONT, color=MUTED,
                                 va="top", zorder=9)
        plate = {"boxstyle": "round,pad=0.3,rounding_size=0.4", "facecolor": SURFACE, "edgecolor": "none"}
        self.end_labels = [
            ax.text(0, 0, pretty_model(s.model), transform=ax.transAxes, color=s.color, size=15, weight="bold",
                    family=FONT, va="center", ha="right", zorder=9, alpha=0, bbox=dict(plate))
            for s in story.series
        ]
        self.factor_labels = [
            ax.text(0, 0, "", color=s.color, size=18, weight="bold", family=FONT, va="center", ha="right", zorder=9)
            for s in story.series
        ]
        self._label_spots = self._place_end_labels()
        self._chart_items = [*ax.get_xticklabels(), *ax.get_yticklabels(), ax.xaxis.label, ax.yaxis.label,
                             *ax.spines.values(), ax.patch]

    def _build_panel(self) -> None:
        p = self.panel = self.fig.add_axes((0.07, 0.065, 0.86, 0.275))
        p.set_xlim(0, 1)
        p.set_ylim(0, 1)
        p.axis("off")

        # Measure: the run, replayed.
        self.ticker_name = p.text(0, 0.84, "", size=26, weight="bold", family=FONT)
        self.ticker_size = p.text(0, 0.68, "", size=20, color=MUTED, family=MONO)
        self.ticker_time = p.text(1, 0.84, "", size=26, weight="bold", family=MONO, ha="right", color=TEXT)
        progress_bg = p.add_patch(Rectangle((0, 0.50), 1, 0.04, color=GRID))
        self.progress = Rectangle((0, 0.50), 0, 0.04, color=ACCENT)
        p.add_patch(self.progress)
        self.counter = p.text(0, 0.36, "", size=16, color=MUTED, family=FONT)
        self.lapse = p.text(0, 0.24, "", size=16, color=MUTED, family=FONT)
        self.measure_items: list = [self.ticker_name, self.ticker_size, self.ticker_time, progress_bg,
                                    self.progress, self.counter, self.lapse]

        # Fit: the leaderboard, one row per class, simplest at the top.
        rows = max(len(self.models), 1)
        row_h = 0.92 / rows
        self.row_y = {m: 0.95 - (i + 0.5) * row_h for i, m in enumerate(self.models)}
        size = 19 if rows <= 7 else 16
        self.row_box = {
            m: p.add_patch(FancyBboxPatch((-0.02, y - row_h * 0.45), 1.04, row_h * 0.9,
                                          boxstyle="round,pad=0,rounding_size=0.015",
                                          facecolor=GRID, edgecolor="none", alpha=0))
            for m, y in self.row_y.items()
        }
        self.row_label = {m: p.text(0.0, y, pretty_model(m), size=size, weight="bold", family=FONT, va="center",
                                    color=TEXT)
                          for m, y in self.row_y.items()}
        self.row_bar = {m: Rectangle((0.32, y - row_h * 0.22), 0, row_h * 0.44, color=FAINT)
                        for m, y in self.row_y.items()}
        for bar in self.row_bar.values():
            p.add_patch(bar)
        self.row_value = {m: p.text(0.32, y, "", size=size - 2, family=MONO, va="center", color=MUTED)
                          for m, y in self.row_y.items()}
        self.row_check = {m: p.text(1.0, y, "", size=size - 2, weight="bold", family=FONT, va="center", ha="right",
                                    color=GOOD)
                          for m, y in self.row_y.items()}

        # Verdict: one card per series.
        shown = min(len(self.story.series), 4)
        self.cards: list[list[Text]] = []
        for i, s in enumerate(self.story.series[:4]):
            y = 0.93 - i * (0.92 / shown)
            self.cards.append([
                p.text(0, y, s.name or "result", size=17, weight="bold", family=FONT, color=s.color, va="top"),
                p.text(0, y - 0.08, pretty_model(s.model), size=30 if shown <= 3 else 24, weight="bold",
                       family=FONT, va="top", color=TEXT),
                p.text(1, y, f"{s.confidence} confidence", size=15, weight="bold", family=FONT, va="top", ha="right",
                       color=CONFIDENCE_COLOR.get(s.confidence, MUTED)),
                p.text(1, y - 0.095, self._card_detail(s), size=12.5, family=FONT, color=MUTED, va="top", ha="right"),
            ])
        self._card_y = [[t.get_position()[1] for t in card] for card in self.cards]

    @staticmethod
    def _card_detail(series: Series) -> str:
        """Why a fit is not rated high, in a few words; otherwise its bound."""
        reasons = [CAVEAT_SHORT.get(c, c) for c in series.caveats]
        return " · ".join(reasons[:2]) if reasons else series.formula

    def _place_end_labels(self) -> list[tuple[float, float]]:
        """Above each line's own end, nudged apart so no two labels overlap."""
        ax = self.ax
        to_axes = ax.transAxes.inverted()
        spots = []
        for i, s in enumerate(self.story.series):
            x, y = s.bound[-1]
            if self.story.log_y and y <= 0:
                y = s.points[-1][1]
            ax_x, ax_y = to_axes.transform(ax.transData.transform((x, y)))
            spots.append([i, ax_x, min(ax_y, 0.97)])
        spots.sort(key=lambda item: item[2])
        for k in range(1, len(spots)):
            if abs(spots[k][1] - spots[k - 1][1]) < 0.3:
                spots[k][2] = max(spots[k][2], spots[k - 1][2] + 0.065)
        placed = [(0.0, 0.0)] * len(spots)
        for i, ax_x, ax_y in spots:
            placed[int(i)] = (min(ax_x, 0.97), min(ax_y + 0.065, 0.96))
        return placed

    def _schedule_trials(self) -> None:
        """When, within the measure act, each trial lands and each median completes.

        A gentle ramp: the first runs arrive one by one, so the eye can follow
        them, and the rest pour in.
        """
        total = len(self.story.trials)
        landing = self.timeline.measure * 0.82
        ramp = [_smooth(0.15 + 0.85 * (k + 1) / total) for k in range(total)]
        low, high = _smooth(0.15), (ramp[-1] if ramp else 1.0)
        self.trial_at = [landing * (r - low) / (high - low) for r in ramp]
        self.median_at = {(t.series, t.x): self.trial_at[k] for k, t in enumerate(self.story.trials)}

    # ---- motion ---------------------------------------------------------

    def act_at(self, t: float) -> tuple[str, float, float]:
        """The act playing at `t`, the seconds into it, and its length."""
        k = min(bisect_right(self.starts, t) - 1, len(self.acts) - 1)
        name, length = self.acts[k]
        return name, t - self.starts[k], length

    def draw(self, t: float) -> None:
        act, local, length = self.act_at(t)
        clock = {name: t - start for (name, _), start in zip(self.acts, self.starts)}
        self.collapse = self._collapse_at(act, clock["verdict"])
        self._draw_header(act, local, length)
        self._draw_hook(act, local, length)
        self._draw_measure(act, clock["measure"])
        self._draw_fit(act, clock["fit"])
        self._draw_verdict(act, clock["verdict"])
        self._draw_panel(act, local, length, clock)
        self._draw_caption(act, local, clock)
        closing = self._fade(local, 0, 0.6) if act == "outro" else 0.0
        self.footer.set_text("measure yours  ·  github.com/djeada/TempoBench" if closing
                             else "made with TempoBench  ·  tembench reel")
        self.footer.set_color(TEXT if closing else FAINT)
        self.footer.set_fontsize(12 + 5 * closing)

    def _collapse_at(self, act: str, local: float) -> float:
        """0 → 1 → 0 across the verdict's collapse beat; 0 when there is none."""
        if self.factors is None or act != "verdict":
            return 0.0
        pace = self.timeline.pace
        fall = _smooth((local - COLLAPSE_START * pace) / (COLLAPSE_FALL * pace))
        rise = _smooth((local - (COLLAPSE_START + COLLAPSE_FALL + COLLAPSE_HOLD) * pace) / (COLLAPSE_BACK * pace))
        return fall * (1 - rise)

    def _draw_header(self, act: str, local: float, length: float) -> None:
        step = {"hook": -1, "measure": 0, "fit": 1, "verdict": 2, "outro": 3}[act]
        for i, (label, fill) in enumerate(zip(self.steps, self.step_fill)):
            label.set_color(TEXT if i == step else (MUTED if i < step else FAINT))
            progress = 1.0 if i < step else (local / length if i == step else 0.0)
            fill.set_width(0.26 * min(1.0, progress))

    def _draw_hook(self, act: str, local: float, length: float) -> None:
        hook = act == "hook"
        out = 1 - _ease((local - length + 0.45 * self.timeline.pace) / (0.45 * self.timeline.pace)) if hook else 0.0
        self.hook_fact.set_alpha(self._fade(local, 0.1, 0.5) * out if hook else 0.0)
        self.hook_question.set_alpha(self._fade(local, 0.9, 0.5) * out if hook else 0.0)
        self.hook_fact.set_y(CHART[1] + CHART[3] / 2 + 0.055 + 0.012 * (1 - self._fade(local, 0.1, 0.5)))
        chart = _ease((local - length + 0.35 * self.timeline.pace) / (0.35 * self.timeline.pace)) if hook else 1.0
        for item in self._chart_items:
            item.set_alpha(chart)
        self.ax.grid(True, color=GRID, linewidth=1, alpha=chart)
        for text in self.legend:
            text.set_alpha(self._fade(local, 0.0, 0.5) if hook else 1.0)

    def _draw_measure(self, act: str, local: float) -> None:
        """Every trial drops into place; a median lands once its size is done."""
        story = self.story
        pace = self.timeline.pace
        offsets: list[list[tuple[float, float]]] = [[] for _ in story.series]
        sizes: list[list[float]] = [[] for _ in story.series]
        started = act != "hook"
        later = act in ("fit", "verdict", "outro")
        for k, trial in enumerate(story.trials):
            age = local - self.trial_at[k]
            if not started or age < 0:
                continue
            fall = 1 - _ease(age / (0.35 * pace))
            y = trial.y * 10 ** (0.35 * fall) if story.log_y else trial.y * (1 + 0.25 * fall)
            offsets[trial.series].append((trial.x, y))
            sizes[trial.series].append(46 * (1 + 1.6 * math.exp(-age / (0.18 * pace))))
        for i, dots in enumerate(self.trial_dots):
            dots.set_offsets(offsets[i] or [(math.nan, math.nan)])
            dots.set_sizes(sizes[i] or [0])
            dots.set_alpha((0.16 if later else 0.55) * (1 - self.collapse))

        for i, s in enumerate(story.series):
            divide = self.factors[i] ** self.collapse if self.factors else 1.0
            done, pops = [], []
            for x, y in s.points:
                age = local - self.median_at.get((i, x), 0.0)
                if started and (age >= 0 or later):
                    done.append((x, y / divide))
                    pops.append(150.0 if later else 150 * (1 + 0.9 * math.exp(-age / (0.15 * pace))))
            self.medians[i].set_offsets(done or [(math.nan, math.nan)])
            self.medians[i].set_sizes(pops or [0])

    def _class_at(self, local: float) -> tuple[int, float]:
        """Which class the fit act is on, and how far the morph into it has got."""
        per = self.timeline.per_class
        j = max(0, min(int(local // per), len(self.models) - 1))
        return j, _smooth((local - j * per) / (per * 0.38))

    def _morphed(self, s: Series, j: int, p: float) -> tuple[list[float], list[float], list[float]]:
        """Series `s` partway from class j-1 to class j: curve xs, ys, and its value at each median."""
        log = self.story.log_y
        cands = s.candidates_by_model
        current = cands.get(self.models[j]) or s.candidates_by_model[s.model]
        previous = (cands.get(self.models[j - 1]) if j > 0 else None) or current
        xs = [x for x, _ in current.curve]
        ys = [_blend(a, b, p, log) for (_, a), (_, b) in zip(previous.curve, current.curve)]
        at = [_blend(a, b, p, log) for a, b in zip(previous.fitted, current.fitted)]
        return xs, ys, at

    def _drawable(self, ys: list[float]) -> list[float]:
        """Drop values a log axis cannot place."""
        return [y if not (self.story.log_y and y <= 0) else math.nan for y in ys]

    def _draw_fit(self, act: str, local: float) -> None:
        story = self.story
        fitting = act == "fit"
        for i, s in enumerate(story.series):
            line, misses = self.fit_lines[i], self.misses[i]
            if act in ("hook", "measure") or not self.models:
                line.set_data([], [])
                misses.set_segments([])
                continue
            if fitting:
                j, p = self._class_at(local)
                xs, ys, at = self._morphed(s, j, p)
                if j == 0:  # the first curve draws itself in
                    keep = max(2, int(len(xs) * p))
                    xs, ys = xs[:keep], ys[:keep]
                tie = 1.0 if j > 0 else self._fade(local, 0.3, 0.3)
                segments = [[(x, y), (x, y + (f - y) * tie)] for (x, y), f in zip(s.points, at)
                            if not (story.log_y and f <= 0)]
                fade = 1.0
            else:  # the verdict: the winning fit gives way to the bound
                winner = s.candidates_by_model[s.model]
                xs, ys = [x for x, _ in winner.curve], [y for _, y in winner.curve]
                segments = []
                fade = 1 - self._fade(local, 0.0, 0.5)
            line.set_data(xs, self._drawable(ys))
            line.set_alpha(fade)
            misses.set_segments(segments)
            misses.set_alpha(0.75 * fade)

        if fitting:
            j, p = self._class_at(local)
            model = self.models[j]
            errors = [s.candidates_by_model[model].error for s in story.series if model in s.candidates_by_model]
            self.class_label.set_text(pretty_model(model) + "?")
            self.class_sub.set_text("misses by " + _percent(sum(errors) / len(errors)) if errors else "")
            self.class_label.set_alpha(_ease(p * 2))
            self.class_sub.set_alpha(_ease(p * 2))
        elif act in ("hook", "measure"):
            self.class_label.set_alpha(0)
            self.class_sub.set_alpha(0)

    def _draw_verdict(self, act: str, local: float) -> None:
        story = self.story
        on = act in ("verdict", "outro")
        collapse = self.collapse
        for i, s in enumerate(story.series):
            line, glow, end, factor = self.bounds[i], self.glows[i], self.end_labels[i], self.factor_labels[i]
            if not on:
                line.set_data([], [])
                glow.set_data([], [])
                end.set_alpha(0)
                if end.get_bbox_patch() is not None:
                    end.get_bbox_patch().set_alpha(0)  # type: ignore[union-attr]
                factor.set_alpha(0)
                continue
            draw_in = 1.0 if act == "outro" else self._fade(local, 0.0, 1.0)
            divide = self.factors[i] ** collapse if self.factors else 1.0
            curve = s.bound[: max(2, int(len(s.bound) * draw_in))]
            xs, ys = [x for x, _ in curve], self._drawable([y / divide for _, y in curve])
            line.set_data(xs, ys)
            glow.set_data(xs, ys)
            glow.set_alpha((0.10 + 0.14 * (0.5 + 0.5 * math.cos(min(local, 3.0) * 2.4))) * (1 - collapse))

            labelled = 0.0 if story.same_class else self._fade(local, 0.9, 0.5) * (1 - collapse)
            end.set_position(self._label_spots[i])
            end.set_alpha(labelled)
            plate = end.get_bbox_patch()
            if plate is not None:
                plate.set_alpha(0.9 * labelled)

            # "÷18" rides down with each line as it falls onto the fastest one.
            if self.factors and self.factors[i] >= 1.15 and collapse > 0:
                fx, fy = s.points[-1]
                factor.set_text(f"÷{_times(self.factors[i])[:-1]}")
                factor.set_position((fx / (1.08 if story.log_x else 1.0), fy / divide * (2.0 if story.log_y else 1.1)))
                factor.set_alpha(min(1.0, collapse * 2))
            else:
                factor.set_alpha(0)

        if on:
            appear = 1.0 if act == "outro" else self._fade(local, 0.9, 0.6)
            if story.same_class:
                self.class_label.set_text(pretty_model(story.series[0].model))
                if collapse > 0.5:
                    sub = "one curve, once the constants go"
                elif len(story.series) > 1:
                    sub = f"for all {len(story.series)}" if len(story.series) > 2 else "for both"
                else:
                    sub = f"{story.series[0].confidence} confidence"
                self.class_sub.set_text(sub)
            else:
                self.class_label.set_text("")
                self.class_sub.set_text("")
            self.class_label.set_alpha(appear)
            self.class_sub.set_alpha(appear)

    def _draw_panel(self, act: str, local: float, length: float, clock: dict[str, float]) -> None:
        story = self.story
        pace = self.timeline.pace

        # Measure: the ticker.
        measuring = act == "measure"
        alpha = 0.0
        if measuring:
            alpha = self._fade(local, 0, 0.3) * (1 - _ease((local - length + 0.3 * pace) / (0.3 * pace)))
        for item in self.measure_items:
            item.set_alpha(alpha)
        if measuring:
            landed = bisect_right(self.trial_at, local)
            if landed:
                trial = story.trials[landed - 1]
                s = story.series[trial.series]
                self.ticker_name.set_text(f"▶  {s.name or 'running'}")
                self.ticker_name.set_color(s.color)
                self.ticker_size.set_text(f"{story.x_name} = {trial.x:,.0f}")
                self.ticker_time.set_text(_ms(trial.y))
            total = len(story.trials)
            self.progress.set_width(landed / total if total else 0)
            self.counter.set_text(f"run {landed} of {total}")
            seconds = story.run_seconds
            if seconds > length * 1.5:
                self.lapse.set_text(f"the real run took {_duration(seconds)}; this is {seconds / length:.0f}× faster")

        # Fit: the leaderboard fills in class by class.
        fitting = act == "fit"
        per = self.timeline.per_class
        winners = {s.model for s in story.series}
        decided_at = per * len(self.models)
        board = (1 - _ease((local - length + 0.3 * pace) / (0.3 * pace))) if fitting else 0.0
        current = self._class_at(local)[0] if fitting else -1
        for k, model in enumerate(self.models):
            start = k * per
            shown = _ease((local - start) / (0.3 * pace)) * board if fitting else 0.0
            errors = [s.candidates_by_model[model].error for s in story.series if model in s.candidates_by_model]
            error = sum(errors) / len(errors) if errors else 1.0
            grow = _ease((local - start - 0.1 * pace) / (per * 0.5)) if fitting else 0.0
            width = 0.52 * math.sqrt(min(error, 1.0)) * grow
            decided = fitting and local >= decided_at
            win = decided and model in winners
            trying = k == current and not decided
            self.row_label[model].set_alpha(shown * (0.6 if decided and not win else 1.0))
            self.row_bar[model].set_width(width)
            self.row_bar[model].set_alpha(shown)
            self.row_bar[model].set_color(GOOD if win else (ACCENT if trying else FAINT))
            self.row_value[model].set_x(0.32 + width + 0.02)
            self.row_value[model].set_text(_percent(error))
            self.row_value[model].set_alpha(shown if grow > 0.05 else 0.0)
            self.row_value[model].set_color(GOOD if win else MUTED)
            self.row_box[model].set_alpha(shown * 0.9 if trying or win else 0.0)
            self.row_check[model].set_text("✓ best fit" if win else "")
            self.row_check[model].set_alpha(_ease((local - decided_at) / (0.3 * pace)) * board if win else 0.0)

        # Verdict: the cards rise in, one after another, once any collapse is over.
        final = act in ("verdict", "outro")
        first = (COLLAPSE_START + COLLAPSE_FALL + COLLAPSE_HOLD + COLLAPSE_BACK if self.factors else 1.2)
        for i, card in enumerate(self.cards):
            appear = 1.0 if act == "outro" else (self._fade(clock["verdict"], first + 0.3 * i, 0.5) if final else 0.0)
            for text, y in zip(card, self._card_y[i]):
                text.set_alpha(appear)
                text.set_y(y - 0.05 * (1 - appear))

    def _draw_caption(self, act: str, local: float, clock: dict[str, float]) -> None:
        story = self.story
        if act == "hook":
            text, alpha = "", 0.0
        elif act == "measure":
            sizes = sorted({x for s in story.series for x, _ in s.points})
            text = (f"{len(sizes)} sizes, {story.x_name} = {human(sizes[0])} → {human(sizes[-1])}. "
                    "Faint dots are runs, solid dots medians.")
            alpha = self._fade(local, 0.2, 0.4)
        elif act == "fit":
            text = "Fit every growth class to the medians. The lines show how far each one misses."
            alpha = self._fade(local, 0, 0.3)
        else:
            if self.factors is not None and act == "verdict" and self.collapse > 0:
                slow = max(range(len(story.series)), key=lambda i: self.factors[i])  # type: ignore[index]
                text = (f"Divide out each one's constant ({_times(self.factors[slow])} for "
                        f"{story.series[slow].name}) and the lines become one. That is what the same "
                        f"{pretty_model(story.series[0].model)} means.")
            else:
                text = self._headline()
            alpha = self._fade(clock["verdict"], 0.6, 0.4) if act == "verdict" else 1.0
        self.caption.set_text(textwrap.fill(text, 44))
        self.caption.set_alpha(alpha)

    def _headline(self) -> str:
        story = self.story
        series = story.series
        model = pretty_model(series[0].model)
        if len(series) == 1:
            return f"{series[0].name or 'It'} grows as {model}."
        if story.same_class:
            everyone = "Both" if len(series) == 2 else f"All {len(series)}"
            line = f"{everyone} grow as {model}."
            if self.gap and self.gap[2] >= 1.5:
                slow, fast, ratio = self.gap
                line += f" {slow.name} is {_times(ratio)} slower than {fast.name}, yet scales the same way."
            return line
        order = {m: k for k, m in enumerate(self.models)}
        groups: dict[str, list[str]] = {}
        for s in sorted(series, key=lambda s: -order.get(s.model, 0)):
            groups.setdefault(s.model, []).append(s.name)
        parts = [f"{' and '.join(names)} {'grow' if len(names) > 1 else 'grows'} as {pretty_model(m)}"
                 for m, names in groups.items()]
        return "; ".join(parts) + "."

    # ---- output ---------------------------------------------------------

    def frames(self, fps: float) -> int:
        return int(math.ceil(self.duration * fps))


def render(
    story: Story,
    output: Path | None,
    *,
    width: int = 1080,
    fps: int = 30,
    poster: Path | None = None,
    timeline: Timeline | None = None,
    audio: bool = True,
    on_frame: Callable[[int, int], None] | None = None,
) -> Reel:
    """Write the reel to `output` (MP4 via ffmpeg) and/or its last frame to `poster`.

    With `audio`, the video gets the synthesised soundtrack (see `audio`).
    """
    dpi = width / FIG_W
    reel = Reel(story, timeline, dpi=dpi)
    if output is not None:
        if not FFMpegWriter.isAvailable():
            raise RuntimeError("ffmpeg is needed to encode the video; install it, or pass only --poster")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as scratch:
            silent = Path(scratch) / "video.mp4" if audio else output
            _encode(reel, silent, fps, dpi, on_frame)
            if audio:
                from .audio import soundtrack, write_wav

                sound = Path(scratch) / "audio.wav"
                write_wav(soundtrack(reel), sound)
                _mux(silent, sound, output)
    if poster is not None:
        poster.parent.mkdir(parents=True, exist_ok=True)
        reel.draw(reel.duration - 0.01)
        reel.fig.savefig(poster, dpi=dpi, facecolor=BG)
    return reel


def _encode(reel: Reel, output: Path, fps: int, dpi: float, on_frame: Callable[[int, int], None] | None) -> None:
    """Draw every frame and pipe it to ffmpeg as H.264."""
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", "-movflags", "+faststart"],
        metadata={"title": reel.story.title, "comment": "made with TempoBench"},
    )
    total = reel.frames(fps)
    with writer.saving(reel.fig, str(output), dpi=dpi):
        for k in range(total):
            reel.draw(k / fps)
            writer.grab_frame(facecolor=BG)
            if on_frame:
                on_frame(k + 1, total)


def _mux(video: Path, sound: Path, output: Path) -> None:
    """Join the silent video and its soundtrack, copying the video stream as is."""
    proc = subprocess.run(
        [rcParams["animation.ffmpeg_path"], "-y", "-loglevel", "error", "-i", str(video), "-i", str(sound),
         "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(output)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not add the soundtrack: {proc.stderr.strip()}")
