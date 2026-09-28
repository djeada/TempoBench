"""Portable subprocess group/session management."""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import psutil

if TYPE_CHECKING:
    from subprocess import Popen

#: Blocks for up to the given number of seconds; True once the child has exited.
ExitWaiter = Callable[[float], bool]


def popen_process_group_kwargs() -> dict[str, Any]:
    """Return Popen kwargs that isolate the child in its own process group."""
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": creationflags} if creationflags else {}
    return {"start_new_session": True}


def _popen_waiter(proc: Popen[Any]) -> ExitWaiter:
    def wait(timeout: float) -> bool:
        try:
            proc.wait(timeout)
        except subprocess.TimeoutExpired:
            return False
        return True

    return wait


def terminate_process_group(
    proc: Popen[Any],
    grace_period_sec: float = 2.0,
    wait_exit: ExitWaiter | None = None,
) -> None:
    """Terminate a subprocess and any children started in its process group.

    `wait_exit` lets a caller that reaps the child itself (to collect its
    resource usage) keep this function from racing it for the exit status.
    """
    wait = wait_exit or _popen_waiter(proc)
    if os.name == "nt":
        _terminate_windows(proc, grace_period_sec, wait)
    else:
        _terminate_posix(proc, grace_period_sec, wait)


def kill_leftover_group(proc: Popen[Any]) -> None:
    """Kill whatever is still running in the child's group after it exited.

    A benchmark that backgrounds a helper and exits would otherwise leave it
    competing for the CPU with every later trial.  Windows has no process
    group to sweep once the parent is gone, so this is POSIX only.
    """
    if os.name == "nt":
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _terminate_posix(proc: Popen[Any], grace_period_sec: float, wait: ExitWaiter) -> None:
    if wait(0):
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except OSError:
        try:
            proc.terminate()
        except OSError:
            return

    if not wait(grace_period_sec):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            try:
                proc.kill()
            except OSError:
                return
        wait(grace_period_sec)


def _terminate_windows(proc: Popen[Any], grace_period_sec: float, wait: ExitWaiter) -> None:
    if wait(0):
        return

    # CTRL_BREAK and TerminateProcess reach only the direct child, so a
    # wrapper script's workers would outlive the timeout.  Collect them first:
    # once the parent is gone they can no longer be found by ancestry.
    try:
        descendants = psutil.Process(proc.pid).children(recursive=True)
    except psutil.Error:
        descendants = []

    ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
    try:
        if ctrl_break is None:
            raise AttributeError
        proc.send_signal(ctrl_break)
    except (AttributeError, OSError, ValueError):
        try:
            proc.terminate()
        except OSError:
            pass

    if not wait(grace_period_sec):
        try:
            proc.kill()
        except OSError:
            pass
        wait(grace_period_sec)

    for child in descendants:
        try:
            child.kill()
        except psutil.Error:
            pass
