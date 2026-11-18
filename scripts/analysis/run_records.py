"""Reading rules shared by the 实验 9-11 analyses: server restarts and crash-affected requests.

The runner (src/omni_research/orchestration/experiment.py) writes a restart's "where" as "<phase> c=<c> rep=<r>" after
a closed-loop level, "<phase> rate=<r>" after an open-loop rate, or "phase <phase>" when a server is found dead at the
end of a phase. After a level in which a server exited, it flags that level's request records "crash_affected" (for
the analysis to drop), restarts the server and runs the level again, appending to the same requests file.
"""

from __future__ import annotations

from typing import Any


def restart_phase(where: str | None) -> str | None:
    words = (where or "").split()
    if not words:
        return None
    return words[1] if words[0] == "phase" and len(words) > 1 else words[0]


def restart_flags(phases: list[str], restarts: list[dict[str, Any]]) -> dict[str, tuple[bool, bool]]:
    """Per phase, in run order: (a server restarted during this phase, this phase is at or after a restart)."""
    where = {restart_phase(r.get("where")) for r in restarts}
    out, after = {}, False
    for name in phases:
        here = name in where
        after = after or here
        out[name] = (here, after)
    return out


def clean(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """The records without crash-affected requests, and how many were dropped."""
    kept = [r for r in records if not r.get("crash_affected")]
    return kept, len(records) - len(kept)
