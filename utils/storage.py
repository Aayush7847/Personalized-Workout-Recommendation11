"""
Lightweight local storage (JSON files) for workout history and feedback.

* No database server needed.
* Writes are atomic (write to a temp file, then replace), so a crash while
  saving cannot leave a half-written file.
* If a file is missing it is created; if it is corrupted it is backed up
  (e.g. workout_history.corrupt-20261002-101500.json) and a fresh empty file is
  started, so the app never crashes because of a bad history file.

Feedback is stored as groundwork for future personalisation. It is NOT used
to retrain the recommender in this version.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = PROJECT_ROOT / "data" / "workout_history.json"
FEEDBACK_PATH = PROJECT_ROOT / "data" / "feedback.json"


class JSONListStore:
    """A JSON file that holds a list of records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.last_warning: str | None = None

    def load(self) -> list[dict]:
        self.last_warning = None
        if not self.path.exists():
            self._write([])
            return []
        try:
            text = self.path.read_text(encoding="utf-8").strip()
            data = json.loads(text) if text else []
            if not isinstance(data, list):
                raise ValueError("top-level JSON value is not a list")
            return [r for r in data if isinstance(r, dict)]
        except (json.JSONDecodeError, ValueError, UnicodeDecodeError, OSError) as exc:
            backup = self.path.with_name(
                f"{self.path.stem}.corrupt-{datetime.now():%Y%m%d-%H%M%S}{self.path.suffix}")
            try:
                os.replace(self.path, backup)
                self.last_warning = (f"{self.path.name} was unreadable ({exc}). It was backed up "
                                     f"as {backup.name} and a new file was started.")
            except OSError:
                self.last_warning = f"{self.path.name} was unreadable and has been reset."
            self._write([])
            return []

    def append(self, record: dict) -> dict:
        records = self.load()
        record = {"id": uuid.uuid4().hex[:10],
                  "timestamp": datetime.now().isoformat(timespec="seconds"), **record}
        records.append(record)
        self._write(records)
        return record

    def clear(self) -> None:
        self._write([])

    def _write(self, records: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(records, fh, indent=2, ensure_ascii=False)
            os.chmod(tmp, 0o644)          # mkstemp creates 0600 files; keep normal permissions
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


def history_store(path: str | Path = HISTORY_PATH) -> JSONListStore:
    return JSONListStore(path)


def feedback_store(path: str | Path = FEEDBACK_PATH) -> JSONListStore:
    return JSONListStore(path)


def workout_to_history_record(profile, workout, filter_stage: str) -> dict:
    """Compact, JSON-serialisable history entry."""
    return {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "goal": profile.goal,
        "target": profile.target_muscle,
        "fitness_level": profile.fitness_level,
        "equipment": profile.equipment,
        "duration": profile.duration,
        "estimated_minutes": workout.total_minutes,
        "estimated_calories": workout.total_calories(profile.weight_kg),
        "variation": workout.variation,
        "filter_stage": filter_stage,
        "exercises": [
            {"name": i.exercise_name, "sets": i.sets, "reps": i.reps,
             "rest_seconds": i.rest_seconds, "match_percent": i.match_percent}
            for i in workout.items
        ],
    }
