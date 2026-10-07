"""`tembench reel`: what the video says, and that every frame of it can be drawn."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from tembench.cli import app
from tembench.complexity import fit_models
from tembench.reel import build_story, pretty_model

SIZES = [1000, 2000, 5000, 10000, 20000, 50000, 100000]


def _summary(langs: dict[str, float] | None = None) -> pd.DataFrame:
    """n·log n in every language, each with its own constant factor."""
    langs = langs or {"cpp": 1e-5, "python": 2e-4}
    rows = [
        {"bench": bench, "n": n, "time_ms_median": c * n * math.log(n), "time_ms_count": 5}
        for bench, c in langs.items()
        for n in SIZES
    ]
    return pd.DataFrame(rows)


def _runs(summary: pd.DataFrame) -> list[dict]:
    # Interleaved the way a run writes them: one benchmark after another.
    return [
        {"status": "ok", "bench": row.bench, "params": {"n": int(row.n)},
         "wall_ms": row.time_ms_median + 20, "reported_ms": row.time_ms_median * f}
        for _, row in summary.iterrows()
        for f in (0.98, 1.0, 1.02)
    ] + [{"status": "timeout", "bench": "cpp", "params": {"n": 10**9}}]


def test_story_agrees_with_the_fits_and_tells_the_punchline():
    summary = _summary()
    story = build_story(summary.assign(time_source="reported"), "n", "time_ms_median", runs=_runs(summary))

    fits = fit_models(summary, "n", "time_ms_median", ["bench"]).set_index("bench")
    assert [s.name for s in story.series] == ["cpp", "python"]
    for s in story.series:
        assert s.model == fits.loc[s.name, "model"] == "O(n log n)"
        # The winner misses least, and every contender is fitted alike.
        assert s.candidates_by_model[s.model].error == min(c.error for c in s.candidates)
        assert len(s.candidates[0].fitted) == len(s.points)
    assert story.same_class and story.log_x and story.log_y
    gap = story.speed_gap()
    assert gap is not None
    slow, fast, ratio = gap
    assert (slow.name, fast.name) == ("python", "cpp") and ratio == pytest.approx(20)


def test_trials_replay_in_run_order_with_the_summarys_timing():
    summary = _summary()
    runs = _runs(summary)
    story = build_story(summary.assign(time_source="reported"), "n", "time_ms_median", runs=runs)
    ok = [r for r in runs if r["status"] == "ok"]
    assert len(story.trials) == len(ok)
    assert [story.series[t.series].name for t in story.trials] == [r["bench"] for r in ok]
    # Self-reported, as the summary was, not wall clock with startup in it.
    assert story.trials[0].y == pytest.approx(ok[0]["reported_ms"])

    walled = build_story(summary.assign(time_source="wall"), "n", "time_ms_median", runs=runs)
    assert walled.trials[0].y == pytest.approx(ok[0]["wall_ms"])


def test_without_trial_records_the_medians_stand_in():
    story = build_story(_summary(), "n", "time_ms_median")
    assert len(story.trials) == 2 * len(SIZES)
    assert [t.x for t in story.trials] == sorted(t.x for t in story.trials)


def test_a_summary_with_nothing_measured_is_refused():
    empty = _summary().assign(time_ms_median=float("nan"))
    with pytest.raises(ValueError, match="no successful measurement"):
        build_story(empty, "n", "time_ms_median")


def test_exponential_classes_read_as_on_screen():
    assert pretty_model("O(n² 2^n)") == "O(n²·2ⁿ)"
    assert pretty_model("O(2^n)") == "O(2ⁿ)"
    assert pretty_model("O(n log n)") == "O(n log n)"


# ---- drawing (needs matplotlib) ----


@pytest.fixture
def reel_module():
    pytest.importorskip("matplotlib")
    from tembench.reel import render

    return render


def test_every_moment_of_the_reel_can_be_drawn(reel_module):
    summary = _summary({"cpp": 1e-5, "rust": 1.1e-5, "python": 2e-4})
    story = build_story(summary.assign(time_source="reported"), "n", "time_ms_median", runs=_runs(summary))
    reel = reel_module.Reel(story, dpi=30)
    acts = [name for name, _ in reel.acts]
    assert acts == ["hook", "measure", "fit", "verdict", "outro"]
    seen = set()
    for k in range(int(reel.duration * 4) + 1):
        t = min(k / 4, reel.duration - 1e-6)
        seen.add(reel.act_at(t)[0])
        reel.draw(t)
    assert seen == set(acts)
    assert "All 3 grow as O(n log n)" in reel.caption.get_text().replace("\n", " ")
    assert "python is 20× slower than cpp" in reel.caption.get_text().replace("\n", " ")
    assert reel.hook_fact.get_text().replace("\n", " ") == "python is 20× slower than cpp."


def test_same_class_series_collapse_onto_one_curve(reel_module):
    # Divided by their constant factors, the three medians coincide.
    summary = _summary({"cpp": 1e-5, "rust": 1.1e-5, "python": 2e-4})
    story = build_story(summary, "n", "time_ms_median")
    assert story.constant_factors() == pytest.approx([1.0, 20.0, 1.1], rel=0.02)
    reel = reel_module.Reel(story, dpi=30)
    verdict = reel.starts[[name for name, _ in reel.acts].index("verdict")]
    pace = reel.timeline.pace
    reel.draw(verdict + (reel_module.COLLAPSE_START + reel_module.COLLAPSE_FALL + 0.5) * pace)
    assert reel.collapse == pytest.approx(1.0)
    tops = [m.get_offsets()[-1][1] for m in reel.medians]
    assert max(tops) / min(tops) == pytest.approx(1.0, abs=0.02)
    reel.draw(reel.duration - 0.01)  # and they spring back for the final frame
    assert reel.collapse == 0.0


def test_different_classes_do_not_collapse():
    rows = [{"impl": impl, "n": n, "time_ms_median": f(n)}
            for impl, f in (("quad", lambda n: 1e-6 * n * n), ("lin", lambda n: 1e-3 * n))
            for n in SIZES]
    assert build_story(pd.DataFrame(rows), "n", "time_ms_median").constant_factors() is None


def test_a_faster_reel_keeps_every_beat(reel_module):
    story = build_story(_summary(), "n", "time_ms_median")
    normal = reel_module.Reel(story, dpi=30)
    quick = reel_module.Reel(story, reel_module.Timeline().faster(2.0), dpi=30)
    assert quick.duration == pytest.approx(normal.duration / 2)
    # The hook's question still appears, halfway through the shorter hook.
    quick.draw(quick.acts[0][1] * 0.6)
    assert quick.hook_question.get_alpha() > 0.5


def test_different_classes_are_grouped_in_the_headline(reel_module):
    rows = [{"impl": impl, "n": n, "time_ms_median": f(n)}
            for impl, f in (("quad", lambda n: 1e-6 * n * n), ("lin", lambda n: 1e-3 * n), ("lin2", lambda n: 2e-3 * n))
            for n in SIZES]
    reel = reel_module.Reel(build_story(pd.DataFrame(rows), "n", "time_ms_median"), dpi=30)
    reel.draw(reel.duration - 0.01)
    text = reel.caption.get_text().replace("\n", " ")
    assert "quad grows as O(n²)" in text and "lin and lin2 grow as O(n)" in text


def _write(tmp_path: Path) -> Path:
    summary = _summary()
    path = tmp_path / "summary.csv"
    summary.to_csv(path, index=False)
    (tmp_path / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in _runs(summary)))
    return path


def test_cli_writes_a_poster(tmp_path: Path, reel_module):
    poster = tmp_path / "poster.png"
    result = CliRunner().invoke(app, [
        "reel", "--summary", str(_write(tmp_path)), "--no-video", "--poster", str(poster), "--width", "270",
    ])
    assert result.exit_code == 0, result.output
    assert poster.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert "42 replayed" in result.output


def test_cli_refuses_to_write_nothing(tmp_path: Path):
    result = CliRunner().invoke(app, ["reel", "--summary", str(_write(tmp_path)), "--no-video"])
    assert result.exit_code == 1
    assert "nothing to write" in result.output


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg encodes the video")
def test_cli_encodes_a_vertical_video(tmp_path: Path, reel_module):
    video = tmp_path / "reel.mp4"
    result = CliRunner().invoke(app, [
        "reel", "--summary", str(_write(tmp_path)), "--output", str(video), "--width", "270", "--fps", "10", "--speed", "4",
    ])
    assert result.exit_code == 0, result.output
    data = video.read_bytes()
    assert b"ftyp" in data[:16] and len(data) > 10_000
    if shutil.which("ffprobe"):
        import subprocess

        streams = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        assert sorted(streams) == ["audio", "video"]


# ---- sound ----


def test_the_soundtrack_is_as_long_as_the_reel_and_clean(reel_module):
    import numpy as np

    from tembench.reel.audio import RATE, soundtrack

    summary = _summary()
    story = build_story(summary.assign(time_source="reported"), "n", "time_ms_median", runs=_runs(summary))
    reel = reel_module.Reel(story, reel_module.Timeline().faster(4.0), dpi=30)
    sound = soundtrack(reel)
    assert sound.shape == (int(reel.duration * RATE), 2)
    peak = float(np.max(np.abs(sound)))
    assert 0.8 < peak <= 0.9  # loud enough, never clipping
    assert float(np.abs(np.diff(sound, axis=0)).max()) < 0.5  # no clicks
    # Something sounds when the first trial lands: the plucks are on the beat.
    landing = reel.starts[1] + reel.trial_at[0]
    before = sound[int((landing - 0.05) * RATE): int(landing * RATE)]
    after = sound[int(landing * RATE): int((landing + 0.05) * RATE)]
    assert np.sqrt(np.mean(after**2)) > np.sqrt(np.mean(before**2))
    assert np.array_equal(sound, soundtrack(reel))  # the same reel, the same sound
