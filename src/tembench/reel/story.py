"""What a reel shows, worked out from a finished run — no drawing here.

A reel retells a benchmark in three acts: the measurements arriving in the
order they were taken, every complexity class being tried against them, and
the class that fits best.  `build_story` gathers everything those acts need,
using the same fits as `fits.csv`, so the video can never disagree with the
report.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import pandas as pd

from ..complexity import fit_models
from ..complexity.fitting import _wls_fit
from ..complexity.models import _MODEL_ORDER, _basis_functions
from ..complexity.selection import _rank_models
from ..plotting._common import (
    PALETTE_DARK,
    SINGLE_SERIES,
    default_series,
    fit_frame,
    label,
    wants_log,
)
from ..summarize import TIME_SOURCE_COL, count_column_for, spread_columns_for

#: Samples per drawn curve: smooth at 1080 px wide.
CURVE_SAMPLES = 80


@dataclass(frozen=True)
class Candidate:
    """One complexity class fitted to one series."""

    model: str
    #: Typical relative miss of the fitted curve: 0.12 means 12%.
    error: float
    curve: tuple[tuple[float, float], ...]
    #: The fitted curve at each measured size, aligned with `Series.points`.
    fitted: tuple[float, ...]


@dataclass(frozen=True)
class Series:
    name: str
    color: str
    #: (size, median duration), by size.
    points: tuple[tuple[float, float], ...]
    #: Every class that competed, simplest first.
    candidates: tuple[Candidate, ...]
    model: str
    confidence: str
    caveats: tuple[str, ...]
    formula: str
    #: The reported upper bound, which sits at or above every point.
    bound: tuple[tuple[float, float], ...]

    @property
    def candidates_by_model(self) -> dict[str, Candidate]:
        return {c.model: c for c in self.candidates}


@dataclass(frozen=True)
class Trial:
    """One successful repetition, in the order the run took it."""

    series: int
    x: float
    y: float


@dataclass(frozen=True)
class Story:
    title: str
    subtitle: str
    x_name: str
    y_label: str
    log_x: bool
    log_y: bool
    series: tuple[Series, ...]
    trials: tuple[Trial, ...]

    @property
    def models(self) -> list[str]:
        """Every class some series tried, simplest first."""
        tried = {c.model for s in self.series for c in s.candidates}
        return [m for m in _MODEL_ORDER if m in tried]

    @property
    def same_class(self) -> bool:
        return len({s.model for s in self.series}) == 1

    def speed_gap(self) -> tuple[Series, Series, float] | None:
        """Slowest and fastest series at the largest size they share, and the ratio.

        The punchline of a cross-language reel: wildly different speed, the
        same growth.  None when there is only one series or no shared size.
        """
        if len(self.series) < 2:
            return None
        shared = set.intersection(*({x for x, _ in s.points} for s in self.series))
        if not shared:
            return None
        top = max(shared)
        at_top = [(dict(s.points)[top], s) for s in self.series]
        (fast_y, fast), (slow_y, slow) = min(at_top, key=_first), max(at_top, key=_first)
        if fast_y <= 0 or slow is fast:
            return None
        return slow, fast, slow_y / fast_y


def _first(pair: tuple[float, Series]) -> float:
    return pair[0]


def pretty_model(model: str) -> str:
    """`O(n² 2^n)` as it reads on screen: `O(n²·2ⁿ)`."""
    return model.replace("² 2^n", "²·2ⁿ").replace("2^n", "2ⁿ")


def _sample_xs(xs: Sequence[float], log: bool) -> list[float]:
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        return [lo]
    if log:
        ratio = (hi / lo) ** (1 / (CURVE_SAMPLES - 1))
        return [lo * ratio**i for i in range(CURVE_SAMPLES)]
    step = (hi - lo) / (CURVE_SAMPLES - 1)
    return [lo + i * step for i in range(CURVE_SAMPLES)]


def _curve(model: str, C: float, baseline: float, xs: Sequence[float]) -> tuple[tuple[float, float], ...]:
    fn = _basis_functions()[model]
    return tuple((x, C * fn(x) + baseline) for x in xs)


def _relative_error(model: str, C: float, baseline: float, x: Sequence[float], y: Sequence[float]) -> float:
    fn = _basis_functions()[model]
    misses = [(yi - (C * fn(xi) + baseline)) / yi for xi, yi in zip(x, y) if yi > 0]
    return math.sqrt(sum(m * m for m in misses) / len(misses)) if misses else 0.0


def _candidates(x: list[float], y: list[float], chosen: str, xs: list[float]) -> tuple[Candidate, ...]:
    """Every class the selection scored, plus O(1) and the winner, fitted alike."""
    _, scores = _rank_models(x, y)
    contenders = {"O(1)", chosen, *scores}
    out = []
    for model in _MODEL_ORDER:
        if model not in contenders:
            continue
        C, baseline, _ = _wls_fit(x, y, _basis_functions()[model])
        out.append(Candidate(
            model,
            _relative_error(model, C, baseline, x, y),
            _curve(model, C, baseline, xs),
            tuple(v for _, v in _curve(model, C, baseline, x)),
        ))
    return tuple(out)


def _series_names(keys: list[tuple], by: list[str]) -> list[str]:
    """Name each series by the grouping columns that actually tell them apart."""
    varying = [i for i, col in enumerate(by) if col != SINGLE_SERIES and len({k[i] for k in keys}) > 1]
    if not varying:
        varying = [i for i, col in enumerate(by) if col != SINGLE_SERIES][:1]
    return [" · ".join(str(k[i]) for i in varying) for k in keys]


def build_story(
    summary: pd.DataFrame,
    x: str,
    y: str,
    color: str | None = None,
    runs: Iterable[Mapping[str, object]] = (),
    title: str | None = None,
    strategy: str = "heuristic",
) -> Story:
    """Gather the measurements, contenders and verdict a reel shows.

    `runs` are the trial records of runs.jsonl, replayed in their file order;
    without them each grid point's median stands in for its trials.
    """
    series_col = default_series(summary, color)
    df, by = fit_frame(summary, x, series_col)
    df = df.assign(**{x: pd.to_numeric(df[x], errors="coerce"), y: pd.to_numeric(df[y], errors="coerce")})
    df = df[df[x].map(math.isfinite) & df[y].map(math.isfinite)]
    if df.empty:
        raise ValueError("the summary has no successful measurement to show")

    fits = fit_models(
        df, x_col=x, y_col=y, by=by, strategy=strategy,
        count_col=count_column_for(y), spread_cols=spread_columns_for(y),
    )
    if fits.empty:
        raise ValueError("no series has the two input sizes a fit needs")
    fit_of = {tuple(row[c] for c in by): row for _, row in fits.iterrows()}

    log_x = wants_log(df[x])
    log_y = wants_log(df[y])
    groups = sorted(
        ((key if isinstance(key, tuple) else (key,), g) for key, g in df.groupby(by, dropna=False)),
        key=lambda item: [str(v) for v in item[0]],
    )
    groups = [(key, g) for key, g in groups if key in fit_of]
    names = _series_names([key for key, _ in groups], by)

    series = []
    for i, ((key, group), name) in enumerate(zip(groups, names)):
        fit = fit_of[key]
        group = group.sort_values(x)
        xs, ys = group[x].astype(float).tolist(), group[y].astype(float).tolist()
        samples = _sample_xs(xs, log_x)
        model = str(fit["model"])
        caveats = str(fit.get("caveats") or "")
        series.append(Series(
            name=name,
            color=PALETTE_DARK[i % len(PALETTE_DARK)],
            points=tuple(zip(xs, ys)),
            candidates=_candidates(xs, ys, model, samples),
            model=model,
            confidence=str(fit["confidence"]),
            caveats=tuple(c for c in caveats.split(",") if c),
            formula=str(fit["formula"]),
            bound=_curve(model, float(fit["C"]), float(fit["baseline"]) + float(fit["offset"]), samples),
        ))

    trials = _replay(runs, df, by, x, y, groups)
    sizes = df[x]
    span = f"{x} = {sizes.min():,.0f} → {sizes.max():,.0f}"
    names_line = " · ".join(s.name for s in series if s.name)
    return Story(
        title=title or "How does it scale?",
        subtitle=" vs ".join(s.name for s in series) if 1 < len(series) <= 4 else (names_line or span),
        x_name=x,
        y_label=label(y),
        log_x=log_x,
        log_y=log_y,
        series=tuple(series),
        trials=tuple(trials),
    )


def _replay(
    runs: Iterable[Mapping[str, object]],
    df: pd.DataFrame,
    by: list[str],
    x: str,
    y: str,
    groups: list[tuple[tuple, pd.DataFrame]],
) -> list[Trial]:
    """The successful trials in run order, timed the way the summary timed them.

    The summary uses the self-reported duration for a grid point only when
    every trial of it reported one; a dot taken from the other reading would
    sit visibly off its own median.
    """
    index = {key: i for i, (key, _) in enumerate(groups)}
    source: dict[tuple, str] = {}
    if TIME_SOURCE_COL in df.columns:
        for _, row in df.iterrows():
            source[(*[row[c] for c in by], float(row[x]))] = str(row[TIME_SOURCE_COL])

    trials: list[Trial] = []
    for rec in runs:
        if rec.get("status") != "ok":
            continue
        params = rec.get("params")
        fields = {"bench": rec.get("bench"), **(params if isinstance(params, Mapping) else {}), SINGLE_SERIES: "all"}
        try:
            key = tuple(fields[c] for c in by)
            size = float(fields[x])  # type: ignore[arg-type]
        except (KeyError, TypeError, ValueError):
            continue
        if key not in index:
            continue
        reported = rec.get("reported_ms")
        use_reported = source.get((*key, size), "wall") == "reported" and reported is not None
        value = reported if use_reported else rec.get("wall_ms")
        if isinstance(value, (int, float)) and value > 0:
            trials.append(Trial(index[key], size, float(value)))
    if trials:
        return trials

    # No trial records: each grid point's median, smallest inputs first.
    return sorted(
        (Trial(i, x_val, y_val) for i, (_, group) in enumerate(groups) for x_val, y_val in zip(group[x], group[y])),
        key=lambda t: (t.x, t.series),
    )
