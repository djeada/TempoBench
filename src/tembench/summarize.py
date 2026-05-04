from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, cast

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

    # Outlier filtering (Tukey)
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    if not include_outliers and not ok.empty and "wall_ms" in ok.columns:
        # Apply per exact grid point (e.g. bench/impl/n), so larger n values are not
        # incorrectly removed as "outliers" relative to smaller input sizes.
        keys = list(group_cols)
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

    def p10(s: pd.Series) -> float:
        return s.quantile(0.1)

    def p90(s: pd.Series) -> float:
        return s.quantile(0.9)

    p10.__name__ = "p10"
    p90.__name__ = "p90"

    agg = {
        "wall_ms": ["median", "mean", "count", p10, p90],
        "peak_rss_mb": ["median", "mean"],
    }
    g = ok.groupby(group_cols, dropna=False).agg(cast(Any, agg))
    flat_columns = cast(Any, g.columns).to_flat_index()
    g.columns = ["_".join(col) for col in flat_columns]
    g = g.reset_index()
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
