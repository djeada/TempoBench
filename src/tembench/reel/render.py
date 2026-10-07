"""Drawing a `Story` as a vertical video, one matplotlib frame at a time.

Every frame is a pure function of time: `Reel.draw(t)` puts each artist where
it belongs at `t` seconds, so any single frame (the poster, a test) can be
rendered without playing the ones before it.
"""

from __future__ import annotations

import math
import textwrap
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from matplotlib.animation import FFMpegWriter
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.text import Text
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullLocator

from .story import Story, pretty_model

BG = "#0e0e0d"
SURFACE = "#181817"
TEXT = "#f2f1ec"
MUTED = "#9a998f"
FAINT = "#5c5b55"
GRID = "#262624"
ACCENT = "#5b9cec"
CONFIDENCE_COLOR = {"high": "#5cc68a", "medium": "#e0a43a", "low": "#f07a72"}

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


@dataclass(frozen=True)
class Timeline:
    """When each act starts and how long it lasts, in seconds."""

    intro: float = 2.0
    measure: float = 7.0
    per_class: float = 0.9
    verdict: float = 5.0
    outro: float = 1.5

    def faster(self, speed: float) -> Timeline:
        """The same reel played `speed` times as fast."""
        return Timeline(*(getattr(self, f) / speed for f in ("intro", "measure", "per_class", "verdict", "outro")))

    def acts(self, classes: int) -> list[tuple[str, float]]:
        return [
            ("intro", self.intro),
            ("measure", self.measure),
            ("fit", self.per_class * (classes + 1)),
            ("verdict", self.verdict),
            ("outro", self.outro),
        ]


def _ease_out(p: float) -> float:
    p = min(1.0, max(0.0, p))
    return 1 - (1 - p) ** 3


def _smooth(p: float) -> float:
    p = min(1.0, max(0.0, p))
    return p * p * (3 - 2 * p)


def _clamp(p: float) -> float:
    return min(1.0, max(0.0, p))


def human(value: float) -> str:
    """Axis numbers a viewer reads at a glance: 1k, 20k, 1M, 0.5, 120."""
    if value == 0:
        return "0"
    magnitude = abs(value)
    for limit, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if magnitude >= limit:
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


def _times(ratio: float) -> str:
    return f"{ratio:.0f}×" if ratio >= 10 else f"{ratio:.1f}×"


class Reel:
    """The figure, its artists, and how they move."""

    def __init__(self, story: Story, timeline: Timeline | None = None, dpi: float = 120.0):
        self.story = story
        self.timeline = timeline or Timeline()
        self.models = story.models
        self.acts = self.timeline.acts(len(self.models))
        self.starts = [0.0]
        for _, length in self.acts:
            self.starts.append(self.starts[-1] + length)
        self.duration = self.starts[-1]

        self.fig = Figure(figsize=(FIG_W, FIG_H), dpi=dpi, facecolor=BG)
        self._build_header()
        self._build_chart()
        self._build_panel()

    # ---- layout ---------------------------------------------------------

    def _text(self, x: float, y: float, s: str, **kw) -> Text:
        kw.setdefault("color", TEXT)
        kw.setdefault("family", FONT)
        return self.fig.text(x, y, s, **kw)

    def _build_header(self) -> None:
        story = self.story
        self.brand = self._text(0.07, 0.955, "TEMPOBENCH", size=13, weight="bold", color=ACCENT)
        self.title = self._text(0.07, 0.912, story.title, size=34, weight="bold", va="baseline")
        self.subtitle = self._text(0.07, 0.882, story.subtitle, size=17, color=MUTED, va="baseline")
        self.steps = [
            self._text(0.07 + 0.29 * i, 0.838, f"{i + 1}  {name}", size=15, weight="bold", color=FAINT)
            for i, name in enumerate(("Measure", "Fit", "Verdict"))
        ]
        self.step_bar = Rectangle((0.07, 0.829), 0.22, 0.003, transform=self.fig.transFigure, color=ACCENT)
        self.fig.add_artist(self.step_bar)
        self.caption = self._text(0.07, 0.392, "", size=17, color=TEXT, va="top", linespacing=1.45)
        self.footer = self._text(0.5, 0.025, "made with TempoBench  ·  tembench reel", size=12, color=FAINT, ha="center")

    def _build_chart(self) -> None:
        story = self.story
        ax = self.ax = self.fig.add_axes((0.15, 0.445, 0.78, 0.36), facecolor=SURFACE)
        xs = [x for s in story.series for x, _ in s.points]
        ys = [y for s in story.series for _, y in s.points]
        trial_ys = [t.y for t in story.trials] or ys
        if story.log_x:
            ax.set_xscale("log")
            ax.set_xlim(min(xs) / 1.35, max(xs) * 1.35)
        else:
            pad = (max(xs) - min(xs)) * 0.06 or 1.0
            ax.set_xlim(min(xs) - pad, max(xs) + pad)
        lo, hi = min(min(ys), min(trial_ys)), max(max(ys), max(trial_ys))
        if story.log_y:
            ax.set_yscale("log")
            ax.set_ylim(lo / 2.2, hi * 2.2)
        else:
            ax.set_ylim(0, hi * 1.15)
        for axis, log in ((ax.xaxis, story.log_x), (ax.yaxis, story.log_y)):
            span = math.log10(axis.get_view_interval()[1] / axis.get_view_interval()[0]) if log else 0
            axis.set_major_locator(LogLocator(10, subs=(1.0, 2.0, 5.0) if span < 2.5 else (1.0,)) if log else MaxNLocator(5))
            axis.set_minor_locator(NullLocator())
            axis.set_major_formatter(FuncFormatter(lambda v, _: human(v)))
        ax.tick_params(colors=MUTED, labelsize=13, length=0, pad=8)
        for side, spine in ax.spines.items():
            spine.set_visible(side in ("left", "bottom"))
            spine.set_color(FAINT)
        ax.grid(True, color=GRID, linewidth=1)
        ax.set_axisbelow(True)
        ax.set_xlabel(f"Input size ({story.x_name})", color=MUTED, size=14, family=FONT, labelpad=10)
        ax.set_ylabel(story.y_label, color=MUTED, size=14, family=FONT, labelpad=10)

        self.trial_dots = [ax.scatter([], [], s=34, color=s.color, alpha=0.5, linewidths=0, zorder=3) for s in story.series]
        self.medians = [
            ax.scatter([], [], s=120, color=s.color, edgecolors=BG, linewidths=2, zorder=5) for s in story.series
        ]
        self.curves = {
            (i, c.model): ax.plot([], [], color=s.color, linewidth=2.5, zorder=4, solid_capstyle="round")[0]
            for i, s in enumerate(story.series)
            for c in s.candidates
        }
        # While a class is tried, each median is tied to its curve: how far
        # the class misses is something to see, not just a number.
        self.misses = [LineCollection([], colors=s.color, linewidths=2, alpha=0.8, zorder=4) for s in story.series]
        for collection in self.misses:
            ax.add_collection(collection)
        self.glows = [ax.plot([], [], color=s.color, linewidth=14, alpha=0.0, zorder=4)[0] for s in story.series]
        self.bounds = [ax.plot([], [], color=s.color, linewidth=4.5, zorder=6, solid_capstyle="round")[0] for s in story.series]
        # One class for everyone is said once, large; otherwise each line is
        # labelled at its end, on a plate so the lines do not run through it.
        self.shared_label = ax.text(
            0.04, 0.95, pretty_model(story.series[0].model), transform=ax.transAxes, size=34,
            weight="bold", family=FONT, color=TEXT, va="top", zorder=8, alpha=0,
        )
        plate = {"boxstyle": "round,pad=0.3,rounding_size=0.4", "facecolor": SURFACE, "edgecolor": "none", "alpha": 0.9}
        self.end_labels = [
            ax.text(0, 0, pretty_model(s.model), color=s.color, size=15, weight="bold", family=FONT,
                    va="center", ha="right", zorder=8, alpha=0, bbox=plate)
            for s in story.series
        ]
        self._label_positions = self._place_end_labels()

        # Median of each grid point appears once its last trial is in.
        self._point_trials = Counter((t.series, t.x) for t in story.trials)

    def _build_panel(self) -> None:
        panel = self.panel = self.fig.add_axes((0.07, 0.065, 0.86, 0.275))
        panel.set_xlim(0, 1)
        panel.set_ylim(0, 1)
        panel.axis("off")
        p = panel

        # Act 1: the run, replayed.
        self.ticker_name = p.text(0, 0.80, "", size=21, weight="bold", family=FONT)
        self.ticker_size = p.text(0, 0.66, "", size=17, color=MUTED, family=MONO)
        self.ticker_time = p.text(1, 0.80, "", size=21, weight="bold", family=MONO, ha="right", color=TEXT)
        self.progress_bg = Rectangle((0, 0.48), 1, 0.035, color=GRID)
        self.progress = Rectangle((0, 0.48), 0, 0.035, color=ACCENT)
        p.add_patch(self.progress_bg)
        p.add_patch(self.progress)
        self.counter = p.text(0, 0.36, "", size=15, color=MUTED, family=FONT)

        # Act 2: the leaderboard.  One row per class, simplest at the top.
        rows = len(self.models)
        self.row_y = {m: 0.90 - (i + 0.5) * (0.88 / max(rows, 1)) for i, m in enumerate(self.models)}
        row_h = 0.88 / max(rows, 1)
        self.board_header = p.text(0.30, 0.97, "misses the medians by", size=13, color=MUTED, family=FONT, va="top")
        self.row_labels = {
            m: p.text(0.0, y, pretty_model(m), size=17, weight="bold", family=FONT, va="center", color=TEXT)
            for m, y in self.row_y.items()
        }
        self.row_marks = {
            m: FancyBboxPatch((-0.015, y - row_h * 0.46), 1.03, row_h * 0.92, boxstyle="round,pad=0,rounding_size=0.02",
                              facecolor="none", edgecolor=CONFIDENCE_COLOR["high"], linewidth=2)
            for m, y in self.row_y.items()
        }
        for mark in self.row_marks.values():
            p.add_patch(mark)
        n_series = len(self.story.series)
        lane = min(row_h * 0.7 / n_series, 0.05)
        self.bars: dict[tuple[int, str], Rectangle] = {}
        self.bar_values: dict[tuple[int, str], Text] = {}
        for i, s in enumerate(self.story.series):
            for c in s.candidates:
                y = self.row_y[c.model] + (n_series / 2 - i - 0.5) * lane
                bar = Rectangle((0.30, y - lane * 0.4), 0, lane * 0.8, color=s.color)
                p.add_patch(bar)
                self.bars[(i, c.model)] = bar
                self.bar_values[(i, c.model)] = p.text(0.30, y, "", size=11 if n_series > 1 else 14, color=s.color,
                                                     family=MONO, va="center")

        # Act 3: one card per series.
        self.cards = []
        for i, s in enumerate(self.story.series[:4]):
            y = 0.86 - i * (0.86 / max(1, min(4, n_series)))
            self.cards.append((
                p.text(0, y, s.name or "result", size=16, weight="bold", family=FONT, color=s.color, va="top"),
                p.text(0, y - 0.075, pretty_model(s.model), size=28, weight="bold", family=FONT, color=TEXT, va="top"),
                p.text(1, y, f"{s.confidence} confidence", size=14, weight="bold", family=FONT, va="top", ha="right",
                       color=CONFIDENCE_COLOR.get(s.confidence, MUTED)),
                p.text(1, y - 0.085, self._card_detail(s), size=12, family=FONT, color=MUTED, va="top", ha="right"),
            ))

    @staticmethod
    def _card_detail(series) -> str:
        """Why a fit is not rated high, in a few words; otherwise its bound."""
        reasons = [CAVEAT_SHORT.get(c, c) for c in series.caveats]
        return " · ".join(reasons[:2]) if reasons else series.formula

    def _place_end_labels(self) -> list[tuple[float, float]]:
        """End-of-line positions for the class labels, nudged apart vertically."""
        ax = self.ax
        to_axes = ax.transAxes.inverted()
        spots = []
        for i, s in enumerate(self.story.series):
            x, y = s.bound[-1]
            if self.story.log_y and y <= 0:
                y = s.points[-1][1]
            ax_x, ax_y = to_axes.transform(ax.transData.transform((x, y)))
            spots.append([i, ax_x, min(ax_y, 0.97)])
        # Above each line's own end, and never on top of another label.
        spots.sort(key=lambda item: item[2])
        for k in range(1, len(spots)):
            if abs(spots[k][1] - spots[k - 1][1]) < 0.3:
                spots[k][2] = max(spots[k][2], spots[k - 1][2] + 0.06)
        placed = [(0.0, 0.0)] * len(spots)
        for i, ax_x, ax_y in spots:
            placed[int(i)] = (min(ax_x, 0.97), min(ax_y + 0.06, 0.96))
        return placed

    # ---- motion ---------------------------------------------------------

    def act_at(self, t: float) -> tuple[str, float, float]:
        """The act playing at `t`, the seconds into it, and its length."""
        k = min(bisect_right(self.starts, t) - 1, len(self.acts) - 1)
        name, length = self.acts[k]
        return name, t - self.starts[k], length

    def draw(self, t: float) -> None:
        act, local, length = self.act_at(t)
        order = [name for name, _ in self.acts]
        stage = order.index(act)
        self._header(act, local)
        self._measure(1.0 if stage > 1 else (local / length if act == "measure" else 0.0), act)
        self._fit(act, local if act == "fit" else (math.inf if stage > 2 else -1.0))
        self._show_misses(act, local)
        self._verdict(local if act == "verdict" else (math.inf if act == "outro" else -1.0))
        self._panel(act, local, length)
        closing = _ease_out(local / 0.6) if act == "outro" else 0.0
        self.footer.set_text("measure yours  ·  github.com/djeada/TempoBench" if closing else
                             "made with TempoBench  ·  tembench reel")
        self.footer.set_color(TEXT if closing else FAINT)
        self.footer.set_fontsize(12 + 4 * closing)

    def _header(self, act: str, local: float) -> None:
        intro = act == "intro"
        reveal = _ease_out(local / 0.8) if intro else 1.0
        self.title.set_alpha(reveal)
        self.title.set_y(0.912 - 0.01 * (1 - reveal))
        self.subtitle.set_alpha(_ease_out((local - 0.3) / 0.8) if intro else 1.0)
        self.ax.patch.set_alpha(_ease_out((local - 0.9) / 0.8) if intro else 1.0)
        chart_alpha = _ease_out((local - 0.9) / 0.8) if intro else 1.0
        for item in [*self.ax.get_xticklabels(), *self.ax.get_yticklabels(), self.ax.xaxis.label, self.ax.yaxis.label]:
            item.set_alpha(chart_alpha)
        for spine in self.ax.spines.values():
            spine.set_alpha(chart_alpha)
        self.ax.grid(True, color=GRID, linewidth=1, alpha=chart_alpha)

        step = {"intro": -1, "measure": 0, "fit": 1, "verdict": 2, "outro": 2}[act]
        for i, text in enumerate(self.steps):
            text.set_color(TEXT if i == step else FAINT)
        self.step_bar.set_alpha(0.0 if step < 0 else 1.0)
        if step >= 0:
            self.step_bar.set_x(0.07 + 0.29 * step)

        story = self.story
        if act == "intro":
            sizes = sorted({x for s in story.series for x, _ in s.points})
            caption = (
                f"{len(sizes)} input sizes, from {story.x_name} = {sizes[0]:,.0f} to {sizes[-1]:,.0f}. "
                "Time each one, then find the curve that explains how the time grows."
            )
        elif act == "measure":
            caption = "Each faint dot is one run. The solid dot is the median of a size's runs."
        elif act == "fit":
            caption = "Try every growth class: fit each one to the medians and see how far it misses."
        else:
            caption = self._headline()
        self.caption.set_text(textwrap.fill(caption, 46))
        self.caption.set_alpha(_ease_out((local - 1.2) / 0.6) if intro else 1.0)

    def _headline(self) -> str:
        story = self.story
        series = story.series
        model = pretty_model(series[0].model)
        if len(series) == 1:
            name = series[0].name or "It"
            return f"{name} grows as {model}."
        gap = story.speed_gap()
        if story.same_class:
            everyone = "Both" if len(series) == 2 else f"All {len(series)}"
            line = f"{everyone} grow as {model}."
            if gap and gap[2] >= 1.5:
                slow, fast, ratio = gap
                line += f" {slow.name} is {_times(ratio)} slower than {fast.name}, yet scales the same way."
            return line
        groups: dict[str, list[str]] = {}
        for s in sorted(series, key=lambda s: -self.models.index(s.model) if s.model in self.models else 0):
            groups.setdefault(s.model, []).append(s.name)
        parts = [f"{' and '.join(names)} {'grow' if len(names) > 1 else 'grows'} as {pretty_model(m)}"
                 for m, names in groups.items()]
        return "; ".join(parts) + "."

    def _measure(self, progress: float, act: str) -> None:
        story = self.story
        trials = story.trials
        shown = len(trials) * _smooth(progress) if progress < 1 else float(len(trials))
        count = int(math.floor(shown))
        seen: Counter = Counter()
        offsets: list[list[tuple[float, float]]] = [[] for _ in story.series]
        sizes: list[list[float]] = [[] for _ in story.series]
        for k, trial in enumerate(trials[:count]):
            seen[(trial.series, trial.x)] += 1
            offsets[trial.series].append((trial.x, trial.y))
            age = shown - k
            sizes[trial.series].append(34 * (1 + 2.0 * math.exp(-age / 2.5)) if progress < 1 else 34)
        dim = act in ("fit", "verdict", "outro")
        for i, dots in enumerate(self.trial_dots):
            dots.set_offsets(offsets[i] or [(math.nan, math.nan)])
            dots.set_sizes(sizes[i] or [0])
            dots.set_alpha(0.18 if dim else 0.5)

        for i, s in enumerate(story.series):
            done = [
                (x, y) for x, y in s.points
                if progress >= 1 or 0 < self._point_trials.get((i, x), 0) <= seen[(i, x)]
            ]
            self.medians[i].set_offsets(done or [(math.nan, math.nan)])
            self.medians[i].set_sizes([120] * max(1, len(done)))

        self._current_trial = trials[count - 1] if count else None
        self._trial_count = count

    def _fit(self, act: str, local: float) -> None:
        per = self.timeline.per_class
        for (i, model), line in self.curves.items():
            j = self.models.index(model)
            start = j * per
            if local < start:
                line.set_data([], [])
                continue
            curve = self._visible(self.story.series[i].candidates_by_model[model].curve)
            grow = _ease_out((local - start) / (per * 0.6))
            n = max(2, int(len(curve) * grow))
            line.set_data([p[0] for p in curve[:n]], [p[1] for p in curve[:n]])
            current = start <= local < start + per
            if act == "verdict" or act == "outro":
                line.set_alpha(0)
            else:
                line.set_alpha(1.0 if current else 0.16)
                line.set_linewidth(3.0 if current else 1.6)

    def _show_misses(self, act: str, local: float) -> None:
        per = self.timeline.per_class
        j = int(local // per) if act == "fit" and local >= 0 else -1
        model = self.models[j] if 0 <= j < len(self.models) else None
        reveal = _ease_out((local - j * per - per * 0.45) / (per * 0.3)) if model else 0.0
        for i, s in enumerate(self.story.series):
            candidate = s.candidates_by_model.get(model) if model else None
            segments = []
            if candidate is not None and reveal > 0:
                for (x, y), fitted in zip(s.points, candidate.fitted):
                    if self.story.log_y and fitted <= 0:
                        continue
                    segments.append([(x, y), (x, y + (fitted - y) * reveal)])
            self.misses[i].set_segments(segments)

    def _verdict(self, local: float) -> None:
        for i, s in enumerate(self.story.series):
            line, glow, end = self.bounds[i], self.glows[i], self.end_labels[i]
            if local < 0:
                for artist in (line, glow):
                    artist.set_data([], [])
                end.set_alpha(0)
                continue
            curve = self._visible(s.bound)
            grow = _ease_out(local / 1.1)
            n = max(2, int(len(curve) * grow))
            xs, ys = [p[0] for p in curve[:n]], [p[1] for p in curve[:n]]
            line.set_data(xs, ys)
            glow.set_data(xs, ys)
            pulse = 0.5 + 0.5 * math.cos(min(local, 3.0) * 2.2) if local < 3 else 1.0
            glow.set_alpha(0.12 + 0.12 * pulse)
            x, y = self._label_positions[i]
            end.set_position((x, y))
            end.set_transform(self.ax.transAxes)
            shown = 0.0 if self.story.same_class else _ease_out((local - 0.9) / 0.5)
            end.set_alpha(shown)
            plate = end.get_bbox_patch()
            if plate is not None:
                plate.set_alpha(0.9 * shown)
        if self.story.same_class:
            self.shared_label.set_alpha(_ease_out((local - 0.9) / 0.6) if local >= 0 else 0.0)

    def _visible(self, curve: tuple[tuple[float, float], ...]) -> list[tuple[float, float]]:
        """Drop the points a log axis cannot draw."""
        if not self.story.log_y:
            return list(curve)
        return [(x, y) if y > 0 else (x, math.nan) for x, y in curve]

    def _panel(self, act: str, local: float, length: float) -> None:
        measuring = act == "measure"
        fitting = act == "fit"
        final = act in ("verdict", "outro")
        story = self.story

        for artist in (self.ticker_name, self.ticker_size, self.ticker_time, self.counter, self.progress, self.progress_bg):
            artist.set_visible(measuring)
        if measuring:
            trial = self._current_trial
            if trial is not None:
                s = story.series[trial.series]
                self.ticker_name.set_text(f"▶  {s.name or 'running'}")
                self.ticker_name.set_color(s.color)
                self.ticker_size.set_text(f"{story.x_name} = {trial.x:,.0f}")
                self.ticker_time.set_text(_ms(trial.y))
            total = len(story.trials)
            self.progress.set_width(self._trial_count / total if total else 0)
            self.counter.set_text(f"run {self._trial_count} of {total}")

        self.board_header.set_visible(fitting)
        per = self.timeline.per_class
        for model, text in self.row_labels.items():
            start = self.models.index(model) * per
            text.set_visible(fitting)
            text.set_alpha(_ease_out((local - start) / 0.3) if fitting else 0)
            mark = self.row_marks[model]
            winners = {s.model for s in story.series}
            mark.set_visible(fitting and model in winners and local >= per * len(self.models))
        for (i, model), bar in self.bars.items():
            start = self.models.index(model) * per
            value = self.bar_values[(i, model)]
            started = fitting and local >= start + 0.15
            bar.set_visible(started)
            value.set_visible(started)
            if not started:
                continue
            error = story.series[i].candidates_by_model[model].error
            width = 0.55 * min(1.0, math.sqrt(error / 1.0)) * _ease_out((local - start - 0.15) / (per * 0.6))
            bar.set_width(max(0.0, width))
            value.set_x(0.30 + width + 0.015)
            value.set_text("" if local < start + 0.15 else ("<1%" if error < 0.01 else f"{error:.0%}"))

        for i, texts in enumerate(self.cards):
            appear = _ease_out((local - 0.5 - 0.35 * i) / 0.6) if act == "verdict" else (1.0 if final else 0.0)
            for text in texts:
                text.set_visible(final)
                text.set_alpha(appear)

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
    on_frame: Callable[[int, int], None] | None = None,
) -> Reel:
    """Write the reel to `output` (MP4 via ffmpeg) and/or its last frame to `poster`."""
    dpi = width / FIG_W
    reel = Reel(story, timeline, dpi=dpi)
    if output is not None:
        if not FFMpegWriter.isAvailable():
            raise RuntimeError("ffmpeg is needed to encode the video; install it, or pass only --poster")
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", "-movflags", "+faststart"],
            metadata={"title": story.title, "comment": "made with TempoBench"},
        )
        total = reel.frames(fps)
        with writer.saving(reel.fig, str(output), dpi=dpi):
            for k in range(total):
                reel.draw(k / fps)
                writer.grab_frame(facecolor=BG)
                if on_frame:
                    on_frame(k + 1, total)
    if poster is not None:
        poster.parent.mkdir(parents=True, exist_ok=True)
        reel.draw(reel.duration - 0.01)
        reel.fig.savefig(poster, dpi=dpi, facecolor=BG)
    return reel

