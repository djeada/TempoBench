"""Portable subprocess group/session management."""

from __future__ import annotations

import os
import signal
import subprocess
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from subprocess import Popen


def popen_process_group_kwargs() -> dict[str, Any]:
    """Return Popen kwargs that isolate the child in its own process group."""
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": creationflags} if creationflags else {}
    return {"start_new_session": True}


def terminate_process_group(proc: Popen[str], grace_period_sec: float = 2.0) -> None:
    """Terminate a subprocess and any children started in its process group."""
    if os.name == "nt":
        _terminate_windows(proc, grace_period_sec)
    else:
        _terminate_posix(proc, grace_period_sec)


def _terminate_posix(proc: Popen[str], grace_period_sec: float) -> None:
    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        try:
            proc.terminate()
        except OSError:
            return

    try:
        proc.wait(grace_period_sec)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            try:
                proc.kill()
            except OSError:
                return
        try:
            proc.wait(grace_period_sec)
        except subprocess.TimeoutExpired:
            pass


def _terminate_windows(proc: Popen[str], grace_period_sec: float) -> None:
    if proc.poll() is not None:
        return

    ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
    try:
        if ctrl_break is None:
            raise AttributeError
        proc.send_signal(ctrl_break)
    except (AttributeError, OSError, ValueError):
        try:
            proc.terminate()
        except OSError:
            return

    try:
        proc.wait(grace_period_sec)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except OSError:
            return
        try:
            proc.wait(grace_period_sec)
        except subprocess.TimeoutExpired:
            pass
