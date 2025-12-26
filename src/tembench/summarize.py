from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with Path(path).open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def summarize_runs(path: Path, include_outliers: bool = False) -> pd.DataFrame:
    rows = read_jsonl(path)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # explode params dict to columns
    if "params" in df.columns:
        params_df = pd.json_normalize(df["params"]).add_prefix("")
        df = pd.concat([df.drop(columns=["params"]), params_df], axis=1)
    # focus on ok runs for medians but keep counts
    group_cols = [c for c in ["bench", "impl", "n"] if c in df.columns]
    if not group_cols:
        # fallback to bench only
        group_cols = [c for c in ["bench"] if c in df.columns]
    # Outlier filtering (Tukey)
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    if not include_outliers and not ok.empty and "wall_ms" in ok.columns:
        # apply per grouping of bench/impl/n (n excluded for per-point filtering)
        keys = [c for c in group_cols if c != "n"]
        if keys:
            # Compute per-group bounds via transform to avoid deprecated GroupBy.apply semantics
            gb = ok.groupby(keys, dropna=False)
            q1 = gb["wall_ms"].transform(lambda s: s.quantile(0.25))
            q3 = gb["wall_ms"].transform(lambda s: s.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (ok["wall_ms"] >= lower) & (ok["wall_ms"] <= upper)
            ok = ok[mask]
    def q(x, p):
        return x.quantile(p)
    agg = {
        "wall_ms": ["median", "mean", "count", (lambda s: q(s, 0.1)), (lambda s: q(s, 0.9))],
        "peak_rss_mb": ["median", "mean"],
    }
    g = ok.groupby(group_cols, dropna=False).agg(agg)
    # flatten columns
    g.columns = [
        "_".join([a for a in col if a]) for col in g.columns.to_flat_index()
    ]
    # Fix lambda names into percentiles
    g = g.rename(columns={
        "wall_ms_<lambda_0>": "wall_ms_p10",
        "wall_ms_<lambda_1>": "wall_ms_p90",
        "peak_rss_mb_median": "peak_rss_mb_median",
        "peak_rss_mb_mean": "peak_rss_mb_mean",
    })
    g = g.reset_index()
    # add counts of failures
    if "status" in df.columns:
        counts = df.groupby(group_cols + ["status"]).size().unstack(fill_value=0).reset_index()
        g = g.merge(counts, on=group_cols, how="left")
    return g
