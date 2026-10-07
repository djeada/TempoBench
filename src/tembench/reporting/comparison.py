"""Comparison helpers and HTML comparison report."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Optional, cast

import pandas as pd

from .. import PROJECT_URL
from ..summarize import TIME_COLUMN_PREFERENCE, TIME_SOURCE_COL, grid_columns
from ..system import get_system_info
from .formatting import (
    _fmt_val,
    _stat_card,
    _table_html,
    column_label,
    format_number,
    format_pct,
)
from .resources import render_page


def _key_columns(current: pd.DataFrame, baseline: pd.DataFrame) -> list[str]:
    """Infer the columns that identify a grid point in both summaries.

    Assuming fixed names would join unrelated rows into a cartesian product for
    any sweep named differently, and silently report the resulting nonsense as
    regressions.
    """
    shared = [c for c in current.columns if c in baseline.columns]
    return grid_columns(shared)


#: Why a row's verdict could not come from its timings.  Each one fails the
#: comparison: CI must not pass a grid point it could not actually check.
MISSING_CURRENT = "missing in current"
NO_TIMING = "no successful trial"
SOURCE_CHANGED = "timing source changed"

#: What each problem means, for readers of the comparison report.
PROBLEM_EXPLANATIONS = {
    MISSING_CURRENT: "the current summary has no row for this grid point at all — "
    "the benchmark, or this input size, was not run.",
    NO_TIMING: "the current run has a row, but no trial succeeded: every one failed, "
    "timed out or was skipped.",
    SOURCE_CHANGED: "one side self-reported its duration and the other measured wall "
    "clock, and there is no wall-clock column to compare on instead.",
}

#: Suffixes of the per-metric columns `compare_summaries` adds.
METRIC_SUFFIXES = ("_current", "_baseline", "_delta", "_delta_pct", "_regression")


def compare_summaries(
    current_csv: Path | pd.DataFrame,
    baseline_csv: Path | pd.DataFrame,
    threshold_pct: float = 5.0,
) -> pd.DataFrame:
    """Compare current results against a baseline."""
    current = current_csv if isinstance(current_csv, pd.DataFrame) else pd.read_csv(current_csv)
    baseline = (
        baseline_csv if isinstance(baseline_csv, pd.DataFrame) else pd.read_csv(baseline_csv)
    )

    group_cols = _key_columns(current, baseline)
    if not group_cols:
        return pd.DataFrame()

    merged = current.merge(
        baseline,
        on=group_cols,
        suffixes=("_current", "_baseline"),
        how="outer",
        indicator=True,
    )
    result_cols = list(group_cols)

    # Only one duration decides pass/fail.  A summary can carry both a
    # self-reported and a wall-clock family, and wall clock includes process
    # startup — letting it raise regressions too would both double-count and
    # turn ordinary startup jitter into failures.
    comparable = [
        m
        for m in TIME_COLUMN_PREFERENCE
        if f"{m}_current" in merged.columns and f"{m}_baseline" in merged.columns
    ]
    if not comparable:
        return pd.DataFrame()
    decisive = comparable[0]

    # A row whose canonical duration came from different sources on the two
    # sides (one self-reported, one wall clock) cannot be compared on it: the
    # gap would be process startup, not a change in the code.  Such rows fall
    # back to wall clock, which both sides always have.
    source_cols = (f"{TIME_SOURCE_COL}_current", f"{TIME_SOURCE_COL}_baseline")
    switched = pd.Series(False, index=merged.index)
    if all(c in merged.columns for c in source_cols):
        switched = (
            merged[source_cols[0]].notna()
            & merged[source_cols[1]].notna()
            & (merged[source_cols[0]] != merged[source_cols[1]])
        )

    for metric in comparable:
        curr_col, base_col = f"{metric}_current", f"{metric}_baseline"
        delta = merged[curr_col] - merged[base_col]
        delta_pct = (delta / merged[base_col] * 100).round(2)
        if metric == decisive and metric.startswith("time_ms"):
            delta = delta.mask(switched)
            delta_pct = delta_pct.mask(switched)
        merged[f"{metric}_delta"] = delta
        merged[f"{metric}_delta_pct"] = delta_pct
        result_cols.extend([curr_col, base_col, f"{metric}_delta", f"{metric}_delta_pct"])

    verdict_pct = merged[f"{decisive}_delta_pct"]
    wall_pct = "wall_ms_median_delta_pct"
    if switched.any() and wall_pct in merged.columns:
        verdict_pct = verdict_pct.where(~switched, merged[wall_pct])
        merged["compared_on"] = decisive.rsplit("_", 1)[0]
        merged.loc[switched, "compared_on"] = "wall_ms"
        result_cols.append("compared_on")

    # A grid point the baseline measured but the current run did not is a
    # failure, not a pass: NaN > threshold is False, so without this a point
    # that stopped working entirely would sail through.  A point the baseline
    # never measured has nothing to regress against.
    measured_before = merged[f"{decisive}_baseline"].notna()
    problem = pd.Series("", index=merged.index)
    problem[merged["_merge"] == "right_only"] = MISSING_CURRENT
    problem[(merged["_merge"] == "both") & merged[f"{decisive}_current"].isna()] = NO_TIMING
    if "compared_on" not in merged.columns:
        # The timing source switched and there is no wall clock to fall back on.
        problem[switched & (problem == "")] = SOURCE_CHANGED
    problem[~measured_before] = ""
    merged["problem"] = problem
    merged[f"{decisive}_regression"] = measured_before & (
        (verdict_pct > threshold_pct) | (problem != "")
    )
    result_cols.extend([f"{decisive}_regression", "problem"])

    for metric in ["peak_rss_mb_median", "peak_rss_mb_mean"]:
        curr_col, base_col = f"{metric}_current", f"{metric}_baseline"
        if curr_col in merged.columns and base_col in merged.columns:
            merged[f"{metric}_delta"] = merged[curr_col] - merged[base_col]
            merged[f"{metric}_delta_pct"] = (
                (merged[curr_col] - merged[base_col]) / merged[base_col] * 100
            ).round(2)
            result_cols.extend(
                [curr_col, base_col, f"{metric}_delta", f"{metric}_delta_pct"]
            )

    result_cols = [c for c in result_cols if c in merged.columns]
    return merged[result_cols].reset_index(drop=True)


def comparison_tally(comparison_df: pd.DataFrame, threshold_pct: float) -> dict[str, int]:
    """Count compared configurations, regressions, and improvements.

    Everything is counted on the single decisive metric.  Summing over every
    `_delta_pct` column would tally one faster configuration once per metric it
    happens to carry, letting a run report more wins than it compared.
    """
    regression_cols = [c for c in comparison_df.columns if c.endswith("_regression")]
    decisive_delta = next(
        (c.replace("_regression", "_delta_pct") for c in regression_cols), None
    )
    improvements = 0
    compared = len(comparison_df)
    if decisive_delta and decisive_delta in comparison_df.columns:
        verdict = comparison_df[decisive_delta]
        if "compared_on" in comparison_df.columns and "wall_ms_median_delta_pct" in comparison_df.columns:
            verdict = verdict.where(
                comparison_df["compared_on"] != "wall_ms",
                comparison_df["wall_ms_median_delta_pct"],
            )
        improvements = int((verdict < -threshold_pct).sum())
        # Rows present on one side only were matched against nothing.
        compared = int(verdict.notna().sum())
    problems = 0
    if "problem" in comparison_df.columns:
        problems = int((comparison_df["problem"].fillna("") != "").sum())

    return {
        "compared": compared,
        "regressions": int(sum(comparison_df[c].sum() for c in regression_cols)),
        "improvements": improvements,
        "unmeasured": problems,
    }


def comparison_keys(comparison_df: pd.DataFrame) -> list[str]:
    """The grid columns identifying each row of a comparison."""
    return [
        c
        for c in comparison_df.columns
        if c not in ("problem", "compared_on") and not str(c).endswith(METRIC_SUFFIXES)
    ]


def decisive_metric(comparison_df: pd.DataFrame) -> str | None:
    """The one duration column the verdicts rest on."""
    return next(
        (c[: -len("_regression")] for c in comparison_df.columns if c.endswith("_regression")),
        None,
    )


def verdicts(comparison_df: pd.DataFrame, threshold_pct: float) -> pd.DataFrame:
    """One verdict per row, with the change, current and baseline it rests on.

    Rows that fell back to wall clock (the timing source switched) are judged,
    and so shown, on wall clock.
    """
    metric = decisive_metric(comparison_df)
    if metric is None:
        return pd.DataFrame(index=comparison_df.index)
    pct = comparison_df[f"{metric}_delta_pct"].copy()
    current = comparison_df[f"{metric}_current"].copy()
    baseline = comparison_df[f"{metric}_baseline"].copy()
    if "compared_on" in comparison_df.columns:
        on_wall = comparison_df["compared_on"] == "wall_ms"
        for values, suffix in ((pct, "_delta_pct"), (current, "_current"), (baseline, "_baseline")):
            column = f"wall_ms_median{suffix}"
            if column in comparison_df.columns:
                values.update(comparison_df.loc[on_wall, column])
    problem = comparison_df.get("problem", pd.Series("", index=comparison_df.index)).fillna("")
    verdict = pd.Series("unchanged", index=comparison_df.index)
    verdict[pct > threshold_pct] = "slower"
    verdict[pct < -threshold_pct] = "faster"
    verdict[baseline.isna()] = "new"
    verdict[baseline.isna() & current.isna()] = "unmeasured"
    verdict[problem != ""] = problem[problem != ""]
    return pd.DataFrame(
        {"verdict": verdict, "delta_pct": pct, "current": current, "baseline": baseline}
    )


VERDICT_TEXT = {
    "slower": "Slower",
    "faster": "Faster",
    "unchanged": "Unchanged",
    "new": "New",
    "unmeasured": "Never measured",
    MISSING_CURRENT: "Missing",
    NO_TIMING: "No timing",
    SOURCE_CHANGED: "Not comparable",
}
_VERDICT_CLASS = {
    "slower": "bad",
    "faster": "good",
    "unchanged": "neutral",
    "new": "neutral",
    "unmeasured": "neutral",
}
#: Verdicts that fail the comparison come first.
_VERDICT_ORDER = {MISSING_CURRENT: 0, NO_TIMING: 0, SOURCE_CHANGED: 0, "slower": 1}
_NA = '<span class="na">—</span>'


def _verdict_table(
    comparison_df: pd.DataFrame, verdict: pd.DataFrame, keys: list[str], metric: str
) -> str:
    """The headline table: what changed and by how much, worst first.

    A grid column with a single value is named once above the table rather
    than repeated on every row, so the verdict stays in view on a phone.
    """
    name, unit = _split_label(column_label(metric))
    shared = [k for k in keys if len(keys) > 1 and comparison_df[k].nunique(dropna=False) == 1]
    keys = [k for k in keys if k not in shared]
    order = (
        pd.DataFrame(
            {
                "rank": verdict["verdict"].map(_VERDICT_ORDER).fillna(2),
                "pct": -verdict["delta_pct"].fillna(0),
            }
        )
        .sort_values(["rank", "pct"], kind="stable")
        .index
    )
    esc = html.escape
    rows = []
    for i in order:
        kind = str(verdict.at[i, "verdict"])
        badge_cls = _VERDICT_CLASS.get(kind, "bad")
        cells = [f"<td>{_fmt_val(comparison_df.at[i, k], k)}</td>" for k in keys]
        cells.append(
            f'<td><span class="badge {badge_cls}">{esc(VERDICT_TEXT.get(kind, kind))}</span></td>'
        )
        pct = cast(float, verdict.at[i, "delta_pct"])
        # Only the decisive change is coloured, and only when it decided.
        tone = {"slower": " bad", "faster": " good"}.get(kind, "")
        cells.append(
            f'<td class="num delta{tone}">{format_pct(pct) if pd.notna(pct) else _NA}</td>'
        )
        for column in ("current", "baseline"):
            value = cast(float, verdict.at[i, column])
            cells.append(f'<td class="num">{format_number(value) if pd.notna(value) else _NA}</td>')
        rows.append(f"<tr>{''.join(cells)}</tr>")
    head = "".join(f"<th>{esc(column_label(k))}</th>" for k in keys)
    head += (
        '<th>Verdict</th><th class="num">Change</th>'
        f'<th class="num">Current{esc(unit)}</th><th class="num">Baseline{esc(unit)}</th>'
    )
    scope = "".join(
        f"; every row has {esc(column_label(k))} = {esc(str(comparison_df[k].iloc[0]))}"
        for k in shared
    )
    return (
        f'<p class="desc">Judged on {esc(name.lower())}, worst first{scope}.</p>'
        '<div class="table-wrap"><table class="data-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _split_label(text: str) -> tuple[str, str]:
    """("Median time", " (ms)") from "Median time (ms)"."""
    if text.endswith(")") and " (" in text:
        cut = text.rindex(" (")
        return text[:cut], text[cut:]
    return text, ""


def _problem_section(comparison_df: pd.DataFrame, keys: list[str]) -> str:
    """The grid points that could not be checked, and why each one fails."""
    if "problem" not in comparison_df.columns:
        return ""
    problems = comparison_df[comparison_df["problem"].fillna("") != ""]
    if problems.empty:
        return ""
    reasons = "".join(
        f"<li><strong>{html.escape(VERDICT_TEXT[p])}</strong> — "
        f"{html.escape(PROBLEM_EXPLANATIONS[p])}</li>"
        for p in sorted(set(problems["problem"]))
        if p in PROBLEM_EXPLANATIONS
    )
    return f"""
    <section class="section callout bad">
      <h2>{len(problems)} grid point(s) could not be checked</h2>
      <p class="desc">The baseline measured these, but the current run produced no
        comparable timing. Each one fails the comparison: a benchmark that stopped
        working must not pass as unchanged.</p>
      <ul class="reasons">{reasons}</ul>
      {_table_html(problems[keys + ["problem"]])}
    </section>"""


def generate_comparison_report(
    comparison_df: pd.DataFrame,
    title: str = "TempoBench Comparison Report",
    threshold_pct: float = 5.0,
    output_path: Optional[Path] = None,
    current_name: str | None = None,
    baseline_name: str | None = None,
) -> str:
    """Generate an HTML comparison report.

    It leads with the verdict, then the configurations that decided it; every
    other metric is folded away below.
    """
    # Imported here: the plotting package imports this one for its labels.
    from ..plotting.comparison import plot_deltas
    from ..plotting.save import chart_sections

    tally = comparison_tally(comparison_df, threshold_pct)
    unmeasured = tally["unmeasured"]
    slower = tally["regressions"] - unmeasured
    keys = comparison_keys(comparison_df)
    metric = decisive_metric(comparison_df)
    verdict = verdicts(comparison_df, threshold_pct)

    failed = tally["regressions"] > 0
    parts = []
    if slower:
        parts.append(f"{slower} slower than the baseline")
    if unmeasured:
        parts.append(f"{unmeasured} could not be checked")
    banner = (
        f"{tally['regressions']} regression(s) detected: " + " + ".join(parts)
        if failed
        else "No regressions detected — every configuration is within the threshold"
    )
    cards = [
        _stat_card(str(tally["compared"]), "Compared"),
        _stat_card(str(slower), "Slower", "err" if slower else "ok"),
        _stat_card(str(tally["improvements"]), "Faster", "ok" if tally["improvements"] else ""),
    ]
    if unmeasured:
        cards.append(_stat_card(str(unmeasured), "Could not be checked", "err"))

    sections = [
        f'<div class="status-banner {"fail" if failed else "pass"}" role="status">'
        f"{html.escape(banner)}</div>",
        f'<section class="section"><div class="stat-grid">{"".join(cards)}</div></section>',
        _problem_section(comparison_df, keys),
    ]
    specs: list[dict] = []
    if metric is not None:
        charts_html, specs = chart_sections(
            [plot_deltas(comparison_df, verdict, keys, threshold_pct)]
        )
        sections.append(charts_html)
        sections.append(
            '<section class="section"><h2>Configurations</h2>'
            f"{_verdict_table(comparison_df, verdict, keys, metric)}</section>"
        )
    sections.append(
        '<section class="section"><details><summary>Every metric, side by side</summary>'
        f"{_table_html(comparison_df)}</details></section>"
    )

    esc = html.escape
    meta = f"Regression threshold {threshold_pct:g}%"
    if current_name and baseline_name:
        meta = f"{esc(current_name)} against baseline {esc(baseline_name)} · {meta}"
    page = render_page(
        title=title,
        kind="Comparison",
        meta=meta,
        body="\n".join(s for s in sections if s),
        specs=specs,
        footer=f'Generated by <a href="{PROJECT_URL}">TempoBench</a> · '
        f"{esc(str(get_system_info()['timestamp']))}",
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(page, encoding="utf-8")

    return page
