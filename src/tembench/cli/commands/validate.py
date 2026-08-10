"""`tembench validate` — check a config before committing to a full sweep."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from ...config import Config, load_config
from ...runner import expand_grid
from ...runner.grid import _run_grid_point, format_cmd
from ...runner.reported import MARKER_NAME
from ..app import app, console, fail, print_heading


def _smallest_point(cfg: Config) -> dict[str, object]:
    """Pick the cheapest grid point, so the probe costs as little as possible."""
    points = expand_grid(cfg.grid)
    if not points:
        return {}
    key = cfg.limits.growth_key
    if key and any(key in p for p in points):
        def sort_key(point: dict[str, object]):
            value = point.get(key)
            return (value is None, value if isinstance(value, (int, float)) else 0)

        return sorted(points, key=sort_key)[0]
    return points[0]


@app.command()
def validate(
    config: Path = typer.Option(
        ..., exists=True, dir_okay=False, help="Path to YAML config"
    ),
    probe: bool = typer.Option(
        True,
        "--probe/--no-probe",
        help="Execute the smallest grid point once to check the command actually runs",
    ),
):
    """Check a config, show the planned sweep, and trial-run the smallest point.

    [bold]Example:[/bold]
        tembench validate --config examples/unique_bench.yaml

    A full sweep can take many minutes; this answers in seconds whether the
    command runs at all and whether the configured metric will be available.
    """
    cfg = load_config(config)  # raises with a specific message on bad config
    points = expand_grid(cfg.grid)
    reps = max(1, cfg.limits.repeats)
    trials = len(cfg.benchmarks) * len(points) * (reps + max(0, cfg.limits.warmups))

    print_heading("Validating Config", config=config, metric=cfg.limits.metric)

    grid_table = Table(title="Sweep", box=None, title_style="bold")
    grid_table.add_column("Axis")
    grid_table.add_column("Values", justify="right")
    grid_table.add_column("Range", overflow="fold")
    for axis, values in cfg.grid.items():
        marker = " [dim](growth key)[/dim]" if axis == cfg.limits.growth_key else ""
        span = f"{values[0]} … {values[-1]}" if len(values) > 1 else str(values[0])
        grid_table.add_row(f"{axis}{marker}", str(len(values)), span)
    console.print(grid_table)
    console.print()
    console.print(
        f"[dim]Plan[/dim]  {len(cfg.benchmarks)} benchmark(s) x {len(points)} grid point(s) "
        f"x {reps} repeat(s) + {max(0, cfg.limits.warmups)} warm-up(s) = "
        f"[bold]{trials}[/bold] process launch(es)"
    )

    if cfg.limits.growth_key and cfg.limits.growth_key not in cfg.grid:
        console.print(
            f"[yellow]![/yellow] limits.growth_key is {cfg.limits.growth_key!r}, which is "
            "not a grid axis — prune_on_timeout will have nothing to prune."
        )

    sizes = len(cfg.grid.get(cfg.limits.growth_key or "", []))
    if 0 < sizes < 4:
        console.print(
            f"[yellow]![/yellow] Only {sizes} input size(s): a complexity fit needs at "
            "least 4 to separate neighbouring classes."
        )

    # A thin protocol is the most common cause of a confidently wrong class: the
    # noise it leaves is consistent across the sweep, so it bends the curve
    # rather than scattering the points, and nothing downstream can detect it.
    if reps < 3:
        console.print(
            f"[yellow]![/yellow] repeats={reps}: a median of {reps} sample(s) is not a "
            "median, and outlier filtering has nothing to work with. Use 5 or more."
        )
    if cfg.limits.warmups == 0:
        console.print(
            "[yellow]![/yellow] warmups=0: the first, cold measurement of each grid "
            "point is kept. Cold allocator and cache state inflate small inputs, "
            "which tilts the whole curve upward."
        )

    point = _smallest_point(cfg)
    console.print()
    for bench in cfg.benchmarks:
        console.print(f"[dim]{bench.name}[/dim]  {format_cmd(bench.cmd, point)}")

    if not probe:
        return

    console.print()
    console.print(f"[bold]Probing the smallest grid point[/bold] {point or '(no grid)'}")
    failures: list[tuple[str, str, str]] = []
    for bench in cfg.benchmarks:
        result = _run_grid_point(
            bench, point, cfg.limits.timeout_sec, 0, 1, 0,
            cfg.limits.rss_poll_interval_sec, cfg.limits.metric,
        )[0]
        if result.status != "ok":
            lines = (result.stderr or result.stdout or "").strip().splitlines()
            failures.append((bench.name, result.status, lines[-1] if lines else ""))
            continue

        detail = f"{result.wall_ms:.1f} ms wall" if result.wall_ms is not None else "ran"
        if result.reported_ms is not None:
            detail += f", {result.reported_ms:.3f} ms self-reported"
        console.print(f"  [green]✓[/green] {bench.name}  [dim]{detail}[/dim]")

        if result.reported_ms is None and cfg.limits.metric == "auto":
            console.print(
                f"    [yellow]![/yellow] No {MARKER_NAME} marker, so wall-clock time will "
                "be used — process startup will be included in every measurement."
            )

    if failures:
        for name, status, message in failures:
            console.print(f"  [red]✗[/red] {name}  [red]{status}[/red]  {message}")
        raise fail(f"{len(failures)} benchmark(s) did not run.")

    console.print()
    console.print("[green bold]✓ Config is runnable[/green bold]")
