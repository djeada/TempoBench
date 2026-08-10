from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tembench.reporting import generate_report
from tembench.runner.provenance import (
    PROVENANCE_FILENAME,
    read_provenance,
    write_provenance,
)


def test_provenance_records_the_hardware_the_benchmark_ran_on(tmp_path: Path):
    write_provenance(tmp_path, seed=7, workers=3)
    prov = json.loads((tmp_path / PROVENANCE_FILENAME).read_text())

    assert prov["seed"] == 7
    assert prov["workers"] == 3
    assert prov["cmdline"]
    system = prov["system"]
    for key in ("platform", "processor", "cpu_count_logical", "memory_total_gb"):
        assert key in system, f"provenance must record {key} to be reproducible"


def test_read_provenance_tolerates_missing_and_corrupt_files(tmp_path: Path):
    assert read_provenance(tmp_path / "absent.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert read_provenance(bad) is None
    listy = tmp_path / "list.json"
    listy.write_text("[1, 2]")
    assert read_provenance(listy) is None


def _summary(tmp_path: Path) -> Path:
    path = tmp_path / "summary.csv"
    pd.DataFrame(
        [{"bench": "b", "impl": "a", "n": 100, "time_ms_median": 1.0}]
    ).to_csv(path, index=False)
    return path


def test_report_describes_the_recorded_machine_not_the_current_one(tmp_path: Path):
    """A report built elsewhere must still describe where the numbers came from."""
    prov = tmp_path / PROVENANCE_FILENAME
    prov.write_text(json.dumps({
        "ts": "2020-01-01T00:00:00+00:00",
        "seed": 99,
        "workers": 4,
        "cwd": "/build/agent",
        "cmdline": "tembench run --config ci.yaml",
        "system": {
            "timestamp": "2020-01-01T00:00:00+00:00",
            "platform": "Fictional-CI-Runner-9000",
            "python_version": "3.11.0",
            "processor": "unobtainium",
            "architecture": "riscv64",
            "cpu_count_logical": 128,
            "cpu_count_physical": 64,
            "memory_total_gb": 512.0,
        },
    }))

    html = generate_report(_summary(tmp_path), provenance_json=prov)

    assert "Fictional-CI-Runner-9000" in html
    assert "unobtainium" in html
    assert "128 logical" in html
    assert "Recorded when the benchmark ran" in html
    # The invocation that produced the measurements is shown too.
    assert "tembench run --config ci.yaml" in html
    assert "/build/agent" in html


def test_report_labels_the_fallback_when_provenance_is_absent(tmp_path: Path):
    html = generate_report(_summary(tmp_path))
    assert "no provenance snapshot was found" in html


def test_report_falls_back_when_provenance_has_no_system_block(tmp_path: Path):
    prov = tmp_path / PROVENANCE_FILENAME
    prov.write_text(json.dumps({"ts": "2020-01-01T00:00:00+00:00", "seed": 1}))
    html = generate_report(_summary(tmp_path), provenance_json=prov)
    assert "no provenance snapshot was found" in html
    # The run metadata it does carry is still worth showing.
    assert "Seed" in html


def test_appending_keeps_the_earlier_snapshots(tmp_path: Path):
    """`--append` grows the results file, so provenance must grow with it."""
    write_provenance(tmp_path, seed=1, workers=1)
    first = json.loads((tmp_path / PROVENANCE_FILENAME).read_text())

    write_provenance(tmp_path, seed=2, workers=4, append=True)
    second = json.loads((tmp_path / PROVENANCE_FILENAME).read_text())

    assert second["seed"] == 2, "the newest run is the top-level record"
    assert [run["seed"] for run in second["previous"]] == [1]
    assert second["previous"][0]["ts"] == first["ts"]

    write_provenance(tmp_path, seed=3, workers=1, append=True)
    third = json.loads((tmp_path / PROVENANCE_FILENAME).read_text())
    assert [run["seed"] for run in third["previous"]] == [1, 2]
    assert "previous" not in third["previous"][1], "history must not nest"


def test_overwriting_a_run_discards_the_history(tmp_path: Path):
    write_provenance(tmp_path, seed=1, workers=1)
    write_provenance(tmp_path, seed=2, workers=1, append=True)
    write_provenance(tmp_path, seed=3, workers=1)
    prov = json.loads((tmp_path / PROVENANCE_FILENAME).read_text())
    assert "previous" not in prov, "a fresh run replaces the results, and the history"


def test_report_says_when_results_combine_several_runs(tmp_path: Path):
    prov = tmp_path / PROVENANCE_FILENAME
    system = {
        "platform": "Linux", "python_version": "3.12.0", "processor": "x86_64",
        "architecture": "x86_64", "cpu_count_logical": 8, "cpu_count_physical": 4,
        "memory_total_gb": 16.0, "hostname": "builder-1",
    }
    prov.write_text(json.dumps({
        "ts": "2020-01-02T00:00:00+00:00", "seed": 2, "system": system,
        "previous": [{"ts": "2020-01-01T00:00:00+00:00", "seed": 1, "system": system}],
    }))
    html = generate_report(_summary(tmp_path), provenance_json=prov)
    assert "combine 2 appended runs" in html
    assert "different" not in html.split("System Information")[1][:600]


def test_report_warns_when_appended_runs_used_different_machines(tmp_path: Path):
    """Timings measured on different hardware cannot be compared to each other."""
    prov = tmp_path / PROVENANCE_FILENAME

    def system(host: str) -> dict:
        return {
            "platform": "Linux", "python_version": "3.12.0", "processor": "x86_64",
            "architecture": "x86_64", "cpu_count_logical": 8, "cpu_count_physical": 4,
            "memory_total_gb": 16.0, "hostname": host,
        }

    prov.write_text(json.dumps({
        "ts": "2020-01-02T00:00:00+00:00", "seed": 2, "system": system("laptop"),
        "previous": [
            {"ts": "2020-01-01T00:00:00+00:00", "seed": 1, "system": system("ci-runner")},
        ],
    }))
    html = generate_report(_summary(tmp_path), provenance_json=prov)
    assert "2 different" in html
    assert "not comparable" in html
