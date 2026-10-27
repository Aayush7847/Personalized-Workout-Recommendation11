"""
Data loading and preprocessing for the exercise dataset.

Pipeline (see `clean_exercise_data`):

1. Schema check          - every required column must exist.
2. Text normalisation    - strip whitespace, unify case/spelling of categories.
3. Category validation   - map known aliases (e.g. "None" -> "No Equipment"),
                           drop rows whose *essential* categories are unknown
                           or missing (an exercise without a muscle group or
                           equipment cannot be recommended safely).
4. Numeric validation    - coerce to numbers; negative / zero / absurd values
                           become NaN and are imputed with the median of the
                           same exercise type (fallback: global median).
5. Missing text          - instructions / safety notes get a safe default.
6. Duplicates            - duplicate IDs and duplicate exercise names are
                           removed (first occurrence kept).
7. Derived features      - goal list, primary goal, timed flag, kcal/min,
                           TF-IDF keyword text.

The function returns the cleaned frame plus a `CleaningReport` so the app and
the notebook can show exactly what was fixed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from model.config import (EXERCISE_EQUIPMENT, EXERCISE_TYPES, FITNESS_LEVELS,
                          GOALS, MUSCLE_GROUPS, REQUIRED_COLUMNS)
from utils.timing import is_timed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "exercises.csv"


class DatasetError(Exception):
    """Raised when the dataset is missing or unusable."""


# Common spelling variants -> canonical values
_ALIASES = {
    "equipment": {
        "none": "No Equipment", "bodyweight": "No Equipment", "body weight": "No Equipment",
        "no equipment": "No Equipment", "dumbbell": "Dumbbells", "dumbbells": "Dumbbells",
        "barbell": "Barbell", "band": "Resistance Band", "bands": "Resistance Band",
        "resistance band": "Resistance Band", "resistance bands": "Resistance Band",
        "cable": "Cable", "cables": "Cable", "machine": "Machine", "bench": "Bench",
        "kettlebell": "Kettlebell", "kettlebells": "Kettlebell", "treadmill": "Treadmill",
        "pull-up bar": "Pull-up Bar", "pullup bar": "Pull-up Bar", "full gym": "Full Gym",
        "gym": "Full Gym",
    },
    "goal": {
        "fitness": "General Fitness", "general fitness": "General Fitness",
        "weight loss": "Weight Loss", "fat loss": "Weight Loss",
        "muscle gain": "Muscle Gain", "hypertrophy": "Muscle Gain",
        "strength": "Strength", "endurance": "Endurance",
    },
    "exercise_type": {t.lower(): t for t in EXERCISE_TYPES},
    "difficulty": {lvl.lower(): lvl for lvl in FITNESS_LEVELS},
    "muscle_group": {m.lower(): m for m in MUSCLE_GROUPS} | {"fullbody": "Full Body",
                                                             "abs": "Core", "legs": "Legs"},
}

# Valid numeric ranges (anything outside is treated as invalid)
_NUMERIC_RANGES = {
    "duration_minutes": (0.5, 60.0),
    "calories_burned": (1, 1000),
    "sets": (1, 10),
    "rest_seconds": (0, 600),
}


@dataclass
class CleaningReport:
    """Summary of everything the preprocessing step changed."""
    rows_in: int = 0
    rows_out: int = 0
    missing_values: dict[str, int] = field(default_factory=dict)
    invalid_numeric_fixed: dict[str, int] = field(default_factory=dict)
    unknown_categories_dropped: int = 0
    duplicates_removed: int = 0
    text_filled: int = 0

    def as_dict(self) -> dict:
        return {
            "Rows loaded": self.rows_in,
            "Rows after cleaning": self.rows_out,
            "Missing values (before cleaning)": sum(self.missing_values.values()),
            "Invalid numeric values repaired": sum(self.invalid_numeric_fixed.values()),
            "Rows dropped (unknown/missing category)": self.unknown_categories_dropped,
            "Duplicate exercises removed": self.duplicates_removed,
            "Missing text fields filled": self.text_filled,
        }


def load_raw_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Read the CSV with friendly errors."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Dataset not found at '{path}'. The file data/exercises.csv "
                           "should be included in the project folder.")
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - depends on file corruption type
        raise DatasetError(f"Could not read the dataset ({exc}).") from exc
    if df.empty:
        raise DatasetError("The dataset file is empty.")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(f"Dataset is missing required columns: {', '.join(missing)}")
    return df


def _normalise_category(series: pd.Series, column: str) -> pd.Series:
    aliases = _ALIASES[column]
    cleaned = series.astype("string").str.strip().str.lower()
    return cleaned.map(lambda v: aliases.get(v, np.nan) if isinstance(v, str) else np.nan)


def _parse_goals(value) -> list[str]:
    """'Muscle Gain|Strength' -> ['Muscle Gain', 'Strength'] (primary first)."""
    if not isinstance(value, str) or not value.strip():
        return []
    out: list[str] = []
    for part in re.split(r"[|,;/]", value):
        canon = _ALIASES["goal"].get(part.strip().lower())
        if canon and canon not in out:
            out.append(canon)
    return out


def _build_tags(row: pd.Series) -> str:
    """Fallback keyword text for rows without a `tags` value."""
    return " ".join(str(x) for x in (row["exercise_name"], row["muscle_group"],
                                     row["equipment"], row["exercise_type"],
                                     row.get("mechanics", ""))).lower()


def clean_exercise_data(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full preprocessing pipeline. Never modifies the input frame."""
    report = CleaningReport(rows_in=len(df))
    df = df.copy()
    report.missing_values = {c: int(v) for c, v in df[REQUIRED_COLUMNS].isna().sum().items() if v}

    # --- 2/3. categorical columns ------------------------------------------------
    for col in ("muscle_group", "equipment", "difficulty", "exercise_type"):
        df[col] = _normalise_category(df[col], col)
    df["goals_list"] = df["goal"].map(_parse_goals)
    df["exercise_name"] = df["exercise_name"].astype("string").str.strip()

    essential_ok = (df[["muscle_group", "equipment", "difficulty", "exercise_type"]].notna().all(axis=1)
                    & df["exercise_name"].notna() & (df["exercise_name"].str.len() > 0)
                    & (df["goals_list"].map(len) > 0))
    report.unknown_categories_dropped = int((~essential_ok).sum())
    df = df.loc[essential_ok].copy()
    df["goal"] = df["goals_list"].map("|".join)
    df["primary_goal"] = df["goals_list"].map(lambda g: g[0])

    # --- 6. duplicates -------------------------------------------------------------
    before = len(df)
    df["exercise_id"] = df["exercise_id"].astype("string").str.strip()
    name_key = df["exercise_name"].str.lower()
    df = df.loc[~name_key.duplicated(keep="first")]
    df = df.loc[~(df["exercise_id"].notna() & df["exercise_id"].duplicated(keep="first"))]
    report.duplicates_removed = before - len(df)
    # Any row still without an ID receives a new unique one
    missing_id = df["exercise_id"].isna() | (df["exercise_id"].str.len() == 0)
    if missing_id.any():
        df.loc[missing_id, "exercise_id"] = [f"EXN{i:03d}" for i in range(1, int(missing_id.sum()) + 1)]

    # --- 4. numeric columns --------------------------------------------------------
    for col, (lo, hi) in _NUMERIC_RANGES.items():
        values = pd.to_numeric(df[col], errors="coerce")
        invalid = values.isna() | (values < lo) | (values > hi)
        report.invalid_numeric_fixed[col] = int(invalid.sum())
        values = values.where(~invalid)
        group_median = values.groupby(df["exercise_type"]).transform("median")
        values = values.fillna(group_median).fillna(values.median())
        df[col] = values
    df["sets"] = df["sets"].round().astype(int)
    df["rest_seconds"] = df["rest_seconds"].round().astype(int)
    df["calories_burned"] = df["calories_burned"].round().astype(int)
    df["duration_minutes"] = df["duration_minutes"].astype(float).round(1)

    # reps is text ("8-10", "30 sec"); missing -> type-appropriate default
    reps = df["reps"].astype("string").str.strip()
    reps_missing = reps.isna() | (reps.str.len() == 0)
    default_reps = np.where(df["exercise_type"].isin(["Cardio", "HIIT", "Mobility"]), "30 sec", "10-12")
    df["reps"] = reps.where(~reps_missing, pd.Series(default_reps, index=df.index)).astype(str)
    report.invalid_numeric_fixed["reps"] = int(reps_missing.sum())

    # --- 5. text columns -----------------------------------------------------------
    filled = 0
    for col, default in (("instructions", "Perform the movement with controlled form through a comfortable range of motion."),
                         ("safety_notes", "Warm up first, use a manageable load and stop if you feel pain.")):
        s = df[col].astype("string").str.strip()
        miss = s.isna() | (s.str.len() == 0)
        filled += int(miss.sum())
        df[col] = s.where(~miss, default).astype(str)
    report.text_filled = filled

    # --- 7. derived features -------------------------------------------------------
    if "mechanics" not in df.columns:
        df["mechanics"] = np.nan
    df["mechanics"] = df["mechanics"].where(df["mechanics"].isin(["Compound", "Isolation"]), "Compound")
    if "movement_pattern" not in df.columns:
        df["movement_pattern"] = "general"
    df["movement_pattern"] = df["movement_pattern"].fillna("general").astype(str)
    if "tags" not in df.columns:
        df["tags"] = np.nan
    tags = df["tags"].astype("string")
    need_tags = tags.isna() | (tags.str.strip().str.len() == 0)
    df["tags"] = tags.astype(object)
    df.loc[need_tags, "tags"] = df.loc[need_tags].apply(_build_tags, axis=1)
    df["tags"] = df["tags"].astype(str)

    df["is_timed"] = df["reps"].map(is_timed)
    df["kcal_per_min"] = (df["calories_burned"] / df["duration_minutes"]).round(3)

    df = df.reset_index(drop=True)
    report.rows_out = len(df)
    return df, report


def load_exercise_data(path: str | Path = DEFAULT_DATASET_PATH) -> tuple[pd.DataFrame, CleaningReport]:
    """Load and clean the dataset in one call (used by app, notebook and tests)."""
    raw = load_raw_dataset(path)
    df, report = clean_exercise_data(raw)
    if df.empty:
        raise DatasetError("No valid exercises remained after cleaning the dataset.")
    return df, report


# Re-export for convenience
__all__ = ["DatasetError", "CleaningReport", "load_raw_dataset", "clean_exercise_data",
           "load_exercise_data", "DEFAULT_DATASET_PATH", "GOALS", "EXERCISE_EQUIPMENT"]
