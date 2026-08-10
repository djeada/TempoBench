from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, cast

import pandas as pd

#: Column holding the duration that summaries, plots and complexity fits use.
TIME_COL = "time_ms"
#: Column naming where each group's canonical duration came from.
TIME_SOURCE_COL = "time_source"

#: Summary columns that may carry a duration, best first.  Summaries written by
#: older versions of TempoBench only have the ``wall_ms_*`` family, so consumers
#: fall through the list rather than assuming a single name.
TIME_COLUMN_PREFERENCE = (
    "time_ms_median",
    "wall_ms_median",
    "time_ms_mean",
    "wall_ms_mean",
)


def preferred_time_column(columns) -> str | None:
    """Return the best available duration column, or None if there is none."""
    available = set(columns)
    return next((c for c in TIME_COLUMN_PREFERENCE if c in available), None)


#: Suffixes that mark a summary column as an aggregated measurement rather than
#: part of the grid point's identity.
METRIC_SUFFIXES = ("_median", "_mean", "_count", "_p10", "_p90")
#: Bare columns that describe a result rather than identify a grid point.
NON_KEY_COLUMNS = frozenset(
    {TIME_SOURCE_COL, "ok", "failed", "timeout", "error", "skipped"}
)


def count_column_for(value_column: str) -> str | None:
    """Name of the column holding how many trials produced `value_column`.

    Summary columns are `<metric>_<aggregate>`, so the sample count for
    `time_ms_median` lives in `time_ms_count`.
    """
    for suffix in METRIC_SUFFIXES:
        if suffix != "_count" and value_column.endswith(suffix):
            return value_column[: -len(suffix)] + "_count"
    return None


def spread_columns_for(value_column: str) -> tuple[str, str] | None:
    """Names of the percentile columns bracketing `value_column`.

    The gap between them says how repeatable the measurement was, which is the
    only evidence in a summary that the machine was busy while it was taken.
    """
    for suffix in METRIC_SUFFIXES:
        if suffix in ("_count", "_p10", "_p90") or not value_column.endswith(suffix):
            continue
        stem = value_column[: -len(suffix)]
        return f"{stem}_p10", f"{stem}_p90"
    return None


def grid_columns(columns) -> list[str]:
    """Return the summary columns that identify a grid point.

    The grid is user-defined, so the sweep axes have to be recovered from the
    data rather than assumed: `n` and `impl` are only what the bundled examples
    happen to be called.
    """
    return [
        str(c)
        for c in columns
        if not str(c).endswith(METRIC_SUFFIXES) and str(c) not in NON_KEY_COLUMNS
    ]


def infer_x_column(df: pd.DataFrame) -> str | None:
    """Guess which grid axis is the input size.

    The input size is the axis a complexity fit is taken over, so it must be
    numeric and is normally the one swept most widely.  A column literally named
    ``n`` wins outright, since that is the documented convention.
    """
    candidates = [c for c in grid_columns(df.columns) if c != "bench"]
    if "n" in candidates:
        return "n"
    numeric = [
        c
        for c in candidates
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1
    ]
    if not numeric:
        return None
    return max(numeric, key=lambda c: df[c].nunique(dropna=True))


def infer_series_column(df: pd.DataFrame, x: str | None) -> str | None:
    """Guess which grid axis separates the series being compared.

    That is whatever is left once the input-size axis is spoken for — the
    variants held side by side at each size.
    """
    candidates = [c for c in grid_columns(df.columns) if c not in {x, "bench"}]
    if "impl" in candidates:
        return "impl"
    varying = [c for c in candidates if df[c].nunique(dropna=True) > 1]
    if not varying:
        return candidates[0] if candidates else None
    # Prefer a categorical axis, then the one with the fewest distinct values:
    # a handful of named variants reads better than a second numeric sweep.
    non_numeric = [c for c in varying if not pd.api.types.is_numeric_dtype(df[c])]
    pool = non_numeric or varying
    return min(pool, key=lambda c: df[c].nunique(dropna=True))


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with Path(path).open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _add_canonical_time(ok: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Attach the canonical duration column and record which metric it came from.

    A command may time its own hot section and print a ``TEMPOBENCH_MS`` marker,
    which excludes process startup from the measurement.  That reading is
    preferred, but only for groups where *every* trial supplied one — silently
    mixing self-reported and wall-clock durations inside a series would put a
    step into the curve that no algorithm produces.
    """
    ok = ok.copy()
    has_wall = "wall_ms" in ok.columns
    has_reported = "reported_ms" in ok.columns

    if not has_wall and not has_reported:
        ok[TIME_COL] = float("nan")
        ok[TIME_SOURCE_COL] = "wall"
        return ok

    wall = (
        pd.to_numeric(ok["wall_ms"], errors="coerce")
        if has_wall
        else pd.Series(float("nan"), index=ok.index)
    )
    if not has_reported or ok.empty:
        ok[TIME_COL] = wall
        ok[TIME_SOURCE_COL] = "wall"
        return ok

    reported = pd.to_numeric(ok["reported_ms"], errors="coerce")
    complete = (
        reported.notna()
        .groupby([ok[c] for c in group_cols], dropna=False)
        .transform("all")
    )
    ok[TIME_COL] = reported.where(complete, wall)
    ok[TIME_SOURCE_COL] = complete.map({True: "reported", False: "wall"})
    return ok


def _drop_outliers(ok: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """Remove Tukey-fence outliers within each exact grid point.

    Fences are computed per group so that larger input sizes are not discarded
    as "outliers" relative to smaller ones.
    """
    if ok.empty or TIME_COL not in ok.columns or not group_cols:
        return ok
    gb = ok.groupby(group_cols, dropna=False)[TIME_COL]
    q1 = gb.transform(lambda s: s.quantile(0.25))
    q3 = gb.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    return ok[(ok[TIME_COL] >= q1 - 1.5 * iqr) & (ok[TIME_COL] <= q3 + 1.5 * iqr)]


def summarize_runs(path: Path, include_outliers: bool = False) -> pd.DataFrame:
    rows = read_jsonl(path)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    param_cols: List[str] = []
    # explode params dict to columns
    if "params" in df.columns:
        params_records = cast(List[dict[str, Any]], df["params"].tolist())
        params_df = pd.json_normalize(params_records)
        param_cols = list(params_df.columns)
        df = pd.concat([df.drop(columns=["params"]), params_df], axis=1)

    # focus on bench + actual grid keys for medians and counts
    group_cols = [c for c in ["bench"] if c in df.columns]
    group_cols.extend(c for c in param_cols if c in df.columns and c not in group_cols)
    if not group_cols:
        df = df.assign(_group_all="all")
        group_cols = ["_group_all"]

    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    ok = _add_canonical_time(ok, group_cols)
    if not include_outliers:
        ok = _drop_outliers(ok, group_cols)

    def p10(s: pd.Series) -> float:
        return s.quantile(0.1)

    def p90(s: pd.Series) -> float:
        return s.quantile(0.9)

    p10.__name__ = "p10"
    p90.__name__ = "p90"

    agg: dict[str, Any] = {
        TIME_COL: ["median", "mean", "count", p10, p90],
        "wall_ms": ["median", "mean", "count", p10, p90],
        "peak_rss_mb": ["median", "mean"],
    }
    agg = {col: how for col, how in agg.items() if col in ok.columns}
    g = ok.groupby(group_cols, dropna=False).agg(cast(Any, agg))
    flat_columns = cast(Any, g.columns).to_flat_index()
    g.columns = ["_".join(col) for col in flat_columns]
    g = g.reset_index()

    if TIME_SOURCE_COL in ok.columns:
        sources = (
            ok.groupby(group_cols, dropna=False)[TIME_SOURCE_COL].first().reset_index()
        )
        g = g.merge(sources, on=group_cols, how="left")

    # add counts of failures
    if "status" in df.columns:
        counts = (
            df.groupby(group_cols + ["status"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        g = g.merge(counts, on=group_cols, how="left")
    if "_group_all" in g.columns:
        g = g.drop(columns=["_group_all"])
    return g
