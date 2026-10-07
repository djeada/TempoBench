"""Main HTML report builder."""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

from .. import PROJECT_URL
from ..runner.provenance import read_provenance
from ..summarize import infer_series_column, infer_x_column, preferred_time_column
from ..system import get_system_info
from .formatting import _stat_card, _table_html, format_number
from .resources import render_page

#: Fit columns worth a reader's attention, in the order the table shows them.
_FIT_COLUMNS = (
    "display_model",
    "model",
    "formula",
    "confidence",
    "confidence_notes",
    "empirical_exponent",
    "exponent_ci_low",
    "exponent_ci_high",
    "runner_up",
    "nobs",
)
_CONFIDENCE_CLASS = {"high": "good", "medium": "warn", "low": "bad"}


def _size(value: float) -> str:
    """An input size: "100,000" rather than "1.00e+05", "6" rather than "6.00"."""
    return f"{int(value):,}" if float(value).is_integer() else format_number(value)


def _run_statistics(rows: list[dict]) -> list[str]:
    """Stat cards counting every trial status in the raw runs.

    A "retried" record is an attempt superseded by a later one of the same
    trial, so it is shown on its own and not counted as a trial or a failure.
    """
    counts = Counter(str(r.get("status", "unknown")) for r in rows)
    retried = counts.pop("retried", 0)
    cards = [
        _stat_card(str(len(rows) - retried), "Total Trials"),
        _stat_card(str(counts.pop("ok", 0)), "Successful", "ok"),
        _stat_card(str(counts.pop("timeout", 0)), "Timeouts", "warn"),
        _stat_card(str(counts.pop("failed", 0)), "Failed", "err"),
    ]
    # Rarer outcomes only get a card when they happened.
    for status, title, variant in (("error", "Errors", "err"), ("skipped", "Skipped", "warn")):
        if counts.get(status):
            cards.append(_stat_card(str(counts.pop(status)), title, variant))
    for status, count in sorted(counts.items()):
        cards.append(_stat_card(str(count), html.escape(status.title()), "warn"))
    if retried:
        cards.append(_stat_card(str(retried), "Retried Attempts"))
    return cards


def _unmeasured_section(df: pd.DataFrame, time_col: str | None) -> str:
    """List the grid points where no trial succeeded, with what went wrong.

    They stay in the summary so a point that stopped working is not mistaken
    for one never measured; charts cannot draw them, so the report names them.
    """
    if time_col is None or df.empty:
        return ""
    missing = df[pd.to_numeric(df[time_col], errors="coerce").isna()]
    if missing.empty:
        return ""
    keep = [
        c
        for c in missing.columns
        if not str(c).endswith(("_median", "_mean", "_count", "_p10", "_p90"))
        and c != "time_source"
    ]
    return f"""
    <section class="section callout warn">
      <h2>Grid Points Without a Measurement</h2>
      <p class="desc">No trial succeeded at these {len(missing)} grid point(s), so they
        have no timing and are left out of the charts and fits. The status counts say why.</p>
      {_table_html(missing[keep])}
    </section>"""


def _series_name(row: pd.Series, by: list[str], single_bench: bool) -> str:
    """How a fitted series is named on its card: its grid values, joined."""
    from ..plotting._common import SINGLE_SERIES

    shown = [c for c in by if c != SINGLE_SERIES and not (single_bench and c == "bench")]
    if not shown:
        shown = [c for c in by if c != SINGLE_SERIES]
    return " · ".join(str(row[c]) for c in shown) or "All measurements"


def _text(value: object) -> str:
    """A cell as text; empty cells read back from a CSV are NaN, not ""."""
    return "" if value is None or (isinstance(value, float) and pd.isna(value)) else str(value)


def _fit_cards(fits: pd.DataFrame, by: list[str], single_bench: bool) -> str:
    """One card per series: the class, how far to trust it, and its bound."""
    esc = html.escape
    cards = []
    for _, row in fits.iterrows():
        klass = str(row.get("display_model", row["model"]))
        confidence = _text(row.get("confidence"))
        caveats = _text(row.get("caveats"))
        rival = row.get("runner_up")
        alternative = ""
        # When a rival class explains the data about as well, it is named: it
        # says which way the answer might go.
        if "ambiguous-class" in caveats.split(",") and isinstance(rival, str) and rival:
            alternative = f'<span class="fit-alt">or {esc(rival)}</span>'
        badge = ""
        if confidence:
            badge = (
                f'<span class="badge {_CONFIDENCE_CLASS.get(confidence, "neutral")}">'
                f"{esc(confidence)} confidence</span>"
            )
        notes = _text(row.get("confidence_notes"))
        cards.append(
            '<div class="fit-card">'
            f'<div class="fit-series">{esc(_series_name(row, by, single_bench))}</div>'
            f'<div class="fit-class">{esc(klass)}{alternative}</div>'
            f"{badge}"
            f'<code class="fit-bound">{esc(str(row["formula"]))}</code>'
            + (f'<p class="fit-notes">{esc(notes[:1].upper() + notes[1:])}</p>' if notes else "")
            + "</div>"
        )
    return f'<div class="fit-grid">{"".join(cards)}</div>'


def _fits_table(fits: pd.DataFrame, by: list[str]) -> str:
    """The fitted classes with their evidence; every raw column in a fold."""
    from ..plotting._common import SINGLE_SERIES

    keys = [c for c in by if c != SINGLE_SERIES and c in fits.columns]
    columns = [c for c in _FIT_COLUMNS if c in fits.columns]
    # The raw class only adds something when the strict label rewrote it.
    if "display_model" in fits.columns and (
        fits["model"].astype(str) == fits["display_model"].astype(str)
    ).all():
        columns.remove("model")
    # A column with nothing in it (no caveats anywhere, say) is left out.
    columns = [c for c in columns if fits[c].replace("", pd.NA).notna().any()]
    raw = fits.drop(columns=[SINGLE_SERIES], errors="ignore")
    return f"""{_table_html(fits[keys + columns], highlight_col="formula")}
      <details><summary>Every fit column</summary>{_table_html(raw)}</details>"""


def _read_fits(fits: Path | pd.DataFrame | None) -> pd.DataFrame | None:
    if fits is None or isinstance(fits, pd.DataFrame):
        return fits
    if not fits.exists():
        return None
    try:
        return pd.read_csv(fits)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def generate_report(
    summary_csv: Path,
    runs_jsonl: Optional[Path] = None,
    fits_csv: Path | pd.DataFrame | None = None,
    title: str = "TempoBench Report",
    output_path: Optional[Path] = None,
    provenance_json: Optional[Path] = None,
    x: str | None = None,
    series: str | None = None,
    complexity_strategy: str = "heuristic",
) -> str:
    """Generate a comprehensive HTML report.

    It leads with the fitted complexity class of every series, then the runtime
    chart (drawn from the summary, with those same fits), the evidence behind
    each class, and the full results.  `x` and `series` default to the axes
    inferred from the summary; fits not given are computed.

    The System Information section describes the machine the benchmark ran on,
    taken from the provenance snapshot written next to the results.  Falling
    back to the current machine is only correct when the report is produced
    where the run happened, so that case is labelled rather than assumed.
    """
    # Imported here: the plotting package imports this one for its labels.
    from ..plotting import fit_frame, fit_runtime, plot_runtime
    from ..plotting._common import default_series, multi_bench
    from ..plotting.save import chart_sections

    df = pd.read_csv(summary_csv)

    recorded = read_provenance(provenance_json) if provenance_json else None
    recorded_system = (recorded or {}).get("system")
    if isinstance(recorded_system, dict) and recorded_system:
        sysinfo = recorded_system
        sysinfo_origin = "Recorded when the benchmark ran"
    else:
        sysinfo = get_system_info()
        sysinfo_origin = "This machine — no provenance snapshot was found"
    esc = html.escape

    # Results built up with `--append` can span several runs, and comparing
    # timings measured on different hardware is meaningless — so say so.
    earlier_runs = (recorded or {}).get("previous") or []
    if isinstance(earlier_runs, list) and earlier_runs:
        hosts = {
            str((run.get("system") or {}).get("hostname", "unknown"))
            for run in earlier_runs
            if isinstance(run, dict)
        }
        hosts.add(str(sysinfo.get("hostname", "unknown")))
        total_runs = len(earlier_runs) + 1
        sysinfo_origin = (
            f"Recorded when the benchmark ran — results combine {total_runs} appended runs"
        )
        if len(hosts) > 1:
            sysinfo_origin += (
                f'. <strong class="warn-text">Those runs used {len(hosts)} different '
                "machines, so their timings are not comparable.</strong>"
            )

    time_col = preferred_time_column(df.columns)
    x = x or infer_x_column(df)
    if series is None and x is not None:
        series = infer_series_column(df, x)
    series = default_series(df, series)

    # ---- Complexity: the headline -------------------------------------------
    fits = _read_fits(fits_csv)
    by: list[str] = []
    if x is not None and time_col is not None and x in df.columns:
        _, by = fit_frame(df, x, series)
        if fits is None:
            fits, by = fit_runtime(
                df, x=x, y=time_col, color=series, complexity_strategy=complexity_strategy
            )
        elif not fits.empty and not set(by) <= set(fits.columns):
            # Fits grouped some other way cannot be drawn on this chart, so
            # the table keys them by their own columns.
            by = [c for c in fits.columns if c in df.columns]
    single_bench = not multi_bench(df)

    complexity_section = ""
    fits_section = ""
    if fits is not None and not fits.empty:
        complexity_section = f"""
    <section class="section">
      <h2>Complexity Classes</h2>
      <p class="desc">The growth class that best explains each series, and how far the
        measurements support it.</p>
      {_fit_cards(fits, by, single_bench)}
    </section>"""
        fits_section = f"""
    <section class="section">
      <h2>Complexity Analysis</h2>
      <p class="desc">Best-fit class per series, selected by AIC over relative-error fits.
        The upper bound satisfies T(n) ≤ C·f(n) + baseline at every measured size.
        <strong>Confidence</strong> states whether the measurements can support the
        class; anything below <em>high</em> lists the caveats that weakened it.</p>
      {_fits_table(fits, by)}
    </section>"""
    elif x is not None and time_col is not None:
        complexity_section = """
    <section class="section">
      <h2>Complexity Classes</h2>
      <p class="empty-msg">No series has the two or more measured input sizes a
        complexity fit needs.</p>
    </section>"""

    # ---- Overview -------------------------------------------------------------
    cards = [_stat_card(str(len(df)), "Configurations")]
    if x is not None and x in df.columns:
        sizes = pd.to_numeric(df[x], errors="coerce").dropna()
        if not sizes.empty:
            cards.append(
                _stat_card(
                    str(sizes.nunique()),
                    "Input Sizes",
                    detail=esc(f"{_size(sizes.min())} – {_size(sizes.max())}"),
                )
            )
    times = pd.to_numeric(df[time_col], errors="coerce").dropna() if time_col else None
    if times is not None and len(times) < len(df):
        cards.append(_stat_card(str(len(df) - len(times)), "Without a Measurement", "err"))
    if "peak_rss_mb_median" in df.columns:
        rss = pd.to_numeric(df["peak_rss_mb_median"], errors="coerce").dropna()
        if not rss.empty:
            cards.append(_stat_card(f"{format_number(rss.max())} MB", "Peak Memory"))
    if runs_jsonl and runs_jsonl.exists():
        rows = []
        with runs_jsonl.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if rows:
            cards.extend(_run_statistics(rows))
    overview_section = f"""
    <section class="section">
      <h2>Overview</h2>
      <div class="stat-grid">{"".join(cards)}</div>
    </section>"""

    # ---- Runtime charts, one per benchmark when each has its own series ----
    chart_section = ""
    specs: list[dict] = []
    if x is not None and time_col is not None and x in df.columns:
        # Fits read from a CSV are only drawn if they carry the curve itself
        # and are grouped as the chart is; otherwise the chart fits its own.
        needed = {*by, "model", "C", "baseline", "offset", "formula"}
        drawable = fits if fits is not None and needed <= set(fits.columns) else None
        benches = (
            sorted(df["bench"].dropna().astype(str).unique())
            if multi_bench(df) and series != "bench"
            else [None]
        )
        charts = [
            plot_runtime(
                df, x=x, y=time_col, color=series, bench=bench,
                fits=drawable, complexity_strategy=complexity_strategy,
                title=f"Runtime: {bench}" if bench else None,
            )
            for bench in benches
        ]
        chart_section, specs = chart_sections(charts)

    unmeasured_section = _unmeasured_section(df, time_col)

    def _si(key: str, label: str) -> str:
        val = sysinfo.get(key, "N/A") or "N/A"
        return _si_value(label, str(val))

    def _si_value(label: str, value: str) -> str:
        return (
            '<div class="sysinfo-row">'
            f'<span class="sysinfo-key">{esc(label)}</span>'
            f'<span class="sysinfo-val">{esc(value)}</span>'
            "</div>"
        )

    cpu_cores = (
        f"{sysinfo.get('cpu_count_physical', 'N/A')} physical / "
        f"{sysinfo.get('cpu_count_logical', 'N/A')} logical"
    )
    memory_total = f"{sysinfo.get('memory_total_gb', 'N/A')} GB"

    run_rows = ""
    if recorded:
        run_rows = "".join(
            _si_value(label, str(recorded[key]))
            for label, key in [
                ("Run At", "ts"),
                ("Seed", "seed"),
                ("Workers", "workers"),
                ("Working Dir", "cwd"),
                ("Invocation", "cmdline"),
            ]
            if key in recorded
        )

    sysinfo_html = f"""
      <p class="desc">{sysinfo_origin}</p>
      <div class="sysinfo-grid">
        <div>
          {_si('platform', 'Platform')}
          {_si('python_version', 'Python')}
          {_si('processor', 'Processor')}
        </div>
        <div>
          {_si_value('CPU Cores', cpu_cores)}
          {_si_value('Memory', memory_total)}
          {_si('architecture', 'Architecture')}
        </div>
      </div>
      {f'<div class="sysinfo-grid"><div>{run_rows}</div></div>' if run_rows else ''}"""

    generated_at = str(get_system_info()["timestamp"])
    body = f"""
    {complexity_section}
    {overview_section}
    {chart_section}
    {fits_section}
    {unmeasured_section}
    <section class="section">
      <h2>Detailed Results</h2>
      {_table_html(df)}
    </section>
    <section class="section">
      <h2>System Information</h2>
      {sysinfo_html}
    </section>"""
    page = render_page(
        title=title,
        kind="Report",
        meta=f"Generated on {esc(generated_at[:10])} from {esc(Path(summary_csv).name)}",
        body=body,
        specs=specs,
        footer=f'Generated by <a href="{PROJECT_URL}">TempoBench</a> · {esc(generated_at)}',
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(page, encoding="utf-8")

    return page
