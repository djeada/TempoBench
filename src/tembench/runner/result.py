"""Typed trial records used by the benchmark runner."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import IO, Literal

TrialStatus = Literal["ok", "failed", "timeout", "error", "skipped"]


@dataclass(frozen=True, slots=True)
class TrialResult(Mapping[str, object]):
    """Single trial outcome with helpers for JSONL serialization."""

    ts: str
    status: TrialStatus
    rc: int | None = None
    wall_ms: float | None = None
    reported_ms: float | None = None
    peak_rss_mb: float | None = None
    stdout: str | None = None
    stderr: str | None = None
    bench: str | None = None
    cmd: str | None = None
    params: dict[str, object] = field(default_factory=dict)
    #: The configured ``limits.metric``.  Both readings are always stored, so
    #: this is what tells a summary which one the user asked for.
    metric: str | None = None
    #: Launches it took to get this outcome; more than 1 only under --retries.
    attempts: int = 1

    def with_context(
        self,
        *,
        bench: str,
        cmd: str,
        params: Mapping[str, object],
        metric: str | None = None,
    ) -> TrialResult:
        """Attach benchmark metadata to a process-level result."""
        return replace(
            self,
            bench=bench,
            cmd=cmd,
            params=dict(params),
            metric=metric if metric is not None else self.metric,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize to the historical JSONL record shape."""
        record: dict[str, object] = {
            "ts": self.ts,
            "status": self.status,
        }
        if self.bench is not None:
            record["bench"] = self.bench
        if self.cmd is not None:
            record["cmd"] = self.cmd
        if self.bench is not None or self.cmd is not None or self.params:
            record["params"] = dict(self.params)
        if self.rc is not None or self.status in {"ok", "failed", "timeout", "error"}:
            record["rc"] = self.rc
        if self.wall_ms is not None:
            record["wall_ms"] = self.wall_ms
        if self.reported_ms is not None:
            record["reported_ms"] = self.reported_ms
        if self.peak_rss_mb is not None:
            record["peak_rss_mb"] = self.peak_rss_mb
        if self.stdout is not None:
            record["stdout"] = self.stdout
        if self.stderr is not None:
            record["stderr"] = self.stderr
        if self.metric is not None:
            record["metric"] = self.metric
        if self.attempts != 1:
            record["attempts"] = self.attempts
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, object]) -> TrialResult:
        """Recreate a TrialResult from a JSON-compatible record."""
        params = record.get("params", {})
        return cls(
            ts=str(record["ts"]),
            status=record["status"],  # type: ignore[arg-type]
            rc=record.get("rc"),  # type: ignore[arg-type]
            wall_ms=record.get("wall_ms"),  # type: ignore[arg-type]
            reported_ms=record.get("reported_ms"),  # type: ignore[arg-type]
            peak_rss_mb=record.get("peak_rss_mb"),  # type: ignore[arg-type]
            stdout=record.get("stdout"),  # type: ignore[arg-type]
            stderr=record.get("stderr"),  # type: ignore[arg-type]
            bench=record.get("bench"),  # type: ignore[arg-type]
            cmd=record.get("cmd"),  # type: ignore[arg-type]
            params=dict(params) if isinstance(params, Mapping) else {},
            metric=record.get("metric"),  # type: ignore[arg-type]
            attempts=int(record.get("attempts", 1)),  # type: ignore[call-overload]
        )

    def __getitem__(self, key: str) -> object:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


#: Progress callback: (bench name, grid point, repetition, repetitions, result).
TrialCallback = Callable[[str, dict[str, object], int, int, TrialResult], None]


class ResultWriter:
    """Append trial records to runs.jsonl as they finish, and report progress."""

    def __init__(self, handle: IO[str], repeats: int, on_trial: TrialCallback | None):
        self._handle = handle
        self._repeats = repeats
        self._on_trial = on_trial

    def emit(self, bench: str, params: dict[str, object], results: list[TrialResult]) -> None:
        for i, rec in enumerate(results):
            # Flushed per record, so a sweep that is interrupted keeps its data.
            self._handle.write(json.dumps(rec.to_dict()) + "\n")
            self._handle.flush()
            if self._on_trial:
                self._on_trial(bench, params, i + 1, self._repeats, rec)
