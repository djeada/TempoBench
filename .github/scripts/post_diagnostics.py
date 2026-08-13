"""Publish smoke-test output as a commit comment when a build fails.

Job logs require an authenticated token to read. Commit comments do not, so a
failure on a platform the maintainer cannot reproduce locally stays diagnosable
from a plain checkout. Runs only on failure, and only posts what the smoke test
already printed to the log.

Delete this step once the platform in question is green -- it exists to make an
invisible failure visible, not to be part of the permanent build.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

DIAGNOSTICS = Path("smoke-diagnostics.txt")
LIMIT = 60000  # GitHub rejects comment bodies over ~65k.


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")
    sha = os.environ.get("GITHUB_SHA")
    runner_os = os.environ.get("RUNNER_OS", "unknown")
    run_url = (
        f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
        f"{repo}/actions/runs/{os.environ.get('GITHUB_RUN_ID', '')}"
    )
    if not (token and repo and sha):
        print("No token or repository in the environment; nothing posted.")
        return 0
    if not DIAGNOSTICS.is_file():
        print(f"{DIAGNOSTICS} does not exist; the smoke test never ran.")
        return 0

    text = DIAGNOSTICS.read_text(encoding="utf-8", errors="replace")
    if len(text) > LIMIT:
        text = text[-LIMIT:]
        text = "(truncated to the last %d characters)\n\n%s" % (LIMIT, text)

    body = (
        f"### Smoke test failed on {runner_os}\n\n"
        f"[Workflow run]({run_url})\n\n"
        f"```\n{text}\n```\n"
    )
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/commits/{sha}/comments",
        data=json.dumps({"body": body}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "tembench-ci",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            print(f"Posted diagnostics: {json.load(response)['html_url']}")
    except urllib.error.HTTPError as exc:
        # Never turn a diagnostic aid into a second failure: the build has
        # already failed and that is the result that matters.
        print(f"Could not post diagnostics ({exc.code}): {exc.read()[:500]!r}")
    except OSError as exc:
        print(f"Could not post diagnostics: {exc!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
