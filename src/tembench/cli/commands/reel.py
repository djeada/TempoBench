"""`tembench reel` — a short vertical video retelling a benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn

from ...summarize import preferred_time_column, read_jsonl
from ..app import (
    BENCH_OPTION,
    STRATEGY_OPTION,
    ComplexityStrategy,
    app,
    console,
    fail,
    load_summary,
    print_artifact,
    print_axes,
    print_heading,
    require_column,
    resolve_axes,
    select_bench,
)


@app.command()
def reel(
    summary: Path = typer.Option(
        Path("artifacts/summary.csv"), exists=True, dir_okay=False, help="Path to summary CSV"
    ),
    runs: Optional[Path] = typer.Option(
        None,
        dir_okay=False,
        help="Trial records to replay in run order (default: runs.jsonl next to the summary)",
    ),
    output: Optional[Path] = typer.Option(
        Path("artifacts/reel.mp4"), "--output", "-o", help="Video to write (MP4, needs ffmpeg)"
    ),
    poster: Optional[Path] = typer.Option(
        None, help="Also save the final frame as an image (PNG), e.g. for a thumbnail"
    ),
    no_video: bool = typer.Option(False, "--no-video", help="Only write --poster; skip the video"),
    no_audio: bool = typer.Option(False, "--no-audio", help="Leave out the synthesised soundtrack"),
    title: str = typer.Option("How does it scale?", help="Headline shown at the top"),
    x: Optional[str] = typer.Option(None, help="Input-size column (default: inferred)"),
    y: Optional[str] = typer.Option(None, help="Duration column (default: time_ms_median)"),
    color: Optional[str] = typer.Option(None, help="Series column (default: inferred)"),
    bench: Optional[str] = BENCH_OPTION,
    width: int = typer.Option(1080, min=270, help="Video width in pixels; the height is 16/9 of it"),
    fps: int = typer.Option(30, min=10, max=60, help="Frames per second"),
    speed: float = typer.Option(1.0, min=0.25, max=4.0, help="Playback speed: 1.5 makes a ~15 s reel"),
    complexity_strategy: ComplexityStrategy = STRATEGY_OPTION,
):
    """Render a ~23 s vertical video: the run replayed, every class tried, the verdict.

    [bold]Example:[/bold]
        tembench reel --summary artifacts/summary.csv --title "Merge sort" --poster artifacts/reel.png

    Made for sharing: 1080x1920 by default, the shape of a phone held upright.
    """
    try:
        from ...reel import build_story
        from ...reel.render import Timeline, render
    except ImportError:
        raise fail(
            "tembench reel needs matplotlib.",
            'Install it with: pip install "tembench[reel]"',
        ) from None

    if no_video and poster is None:
        raise fail("--no-video leaves nothing to write; pass --poster as well.")
    df = select_bench(load_summary(summary), bench)
    explicit_axes = x is not None and color is not None
    x, color = resolve_axes(df, x, color)
    y = y or preferred_time_column(df.columns)
    if y is None:
        raise fail(f"{summary} has no duration column to show.")
    require_column(df, y, "--y")

    runs = runs or (summary.parent / "runs.jsonl")
    records = read_jsonl(runs) if runs.exists() else []
    if bench is not None:
        records = [r for r in records if r.get("bench") == bench]
    try:
        story = build_story(
            df, x, y, color, records, title=title, strategy=complexity_strategy.value
        )
    except ValueError as e:
        raise fail(f"Nothing to show: {e}.") from None

    video = None if no_video else output
    print_heading(
        "Rendering Reel",
        summary=summary,
        trials=f"{len(story.trials)} replayed" + ("" if records else " (medians; no runs.jsonl found)"),
        series=", ".join(s.name for s in story.series) or "1",
    )
    print_axes(x, color, explicit_axes)
    width -= width % 2  # yuv420p video needs even dimensions
    try:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Drawing frames", total=None)

            def on_frame(done: int, total: int) -> None:
                progress.update(task, completed=done, total=total)

            result = render(
                story, video, width=width, fps=fps, poster=poster,
                timeline=Timeline().faster(speed), audio=not no_audio, on_frame=on_frame,
            )
    except RuntimeError as e:
        raise fail(str(e)) from None

    for s in story.series:
        console.print(f"  [bold]{s.name or 'series'}[/bold]  {s.model}  [dim]{s.confidence} confidence[/dim]")
    console.print()
    if video is not None:
        print_artifact(f"Reel ({result.duration:.0f} s)", video)
    if poster is not None:
        print_artifact("Poster frame", poster)
