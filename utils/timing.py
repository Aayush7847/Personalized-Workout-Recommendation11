"""
Timing helpers shared by the dataset generator and the workout generator.

Keeping the duration formula in one place guarantees that the
`duration_minutes` column of the dataset and the workout-time estimate shown
in the app are computed in exactly the same way.
"""
from __future__ import annotations

import re

from model.config import SECONDS_PER_REP, TRANSITION_SECONDS

_TIME_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(sec|s|min|m)\b", re.IGNORECASE)
_RANGE_RE = re.compile(r"(\d+)\s*-\s*(\d+)")
_NUM_RE = re.compile(r"(\d+)")


def is_timed(reps: str) -> bool:
    """True if the reps field describes a time (e.g. '30 sec', '10 min')."""
    return bool(_TIME_RE.search(str(reps)))


def work_seconds_per_set(reps: str) -> float:
    """
    Seconds of work in one set.

    * '30 sec' -> 30, '10 min' -> 600
    * '8-12'   -> mean reps (10) x SECONDS_PER_REP
    * '12'     -> 12 x SECONDS_PER_REP
    Unparseable values fall back to 10 reps.
    """
    text = str(reps).strip()
    m = _TIME_RE.search(text)
    if m:
        value = float(m.group(1))
        unit = m.group(2).lower()
        return value * 60.0 if unit.startswith("m") else value
    m = _RANGE_RE.search(text)
    if m:
        reps_mean = (int(m.group(1)) + int(m.group(2))) / 2.0
        return reps_mean * SECONDS_PER_REP
    m = _NUM_RE.search(text)
    reps_n = int(m.group(1)) if m else 10
    return reps_n * SECONDS_PER_REP


def estimate_exercise_seconds(sets: int, reps: str, rest_seconds: int) -> float:
    """
    Estimated time for one exercise block:

        sets x work + (sets - 1) x rest + transition
    """
    sets = max(int(sets), 1)
    rest_seconds = max(int(rest_seconds), 0)
    return sets * work_seconds_per_set(reps) + (sets - 1) * rest_seconds + TRANSITION_SECONDS
