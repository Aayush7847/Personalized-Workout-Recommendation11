"""
Automated checks for the Personalized Workout Recommendation System.

Run from the project root with either:
    python tests/test_project.py
    python -m pytest tests            (if pytest is installed)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.config import (DURATIONS, EQUIPMENT_ACCESS, EXERCISE_COUNT_RANGE,  # noqa: E402
                          FITNESS_LEVELS, GOALS, REQUIRED_COLUMNS,
                          TARGET_MUSCLES, USER_EQUIPMENT, USER_EXERCISE_TYPES)
from model.recommender import BLOCK_ORDER, WorkoutRecommender  # noqa: E402
from model.user_profile import UserProfile, build_profile  # noqa: E402
from utils.explanations import explain_exercise  # noqa: E402
from utils.preprocessing import clean_exercise_data, load_raw_dataset  # noqa: E402
from utils.storage import JSONListStore  # noqa: E402
from utils.workout_generator import generate_workout, recommend_and_generate  # noqa: E402

REC = WorkoutRecommender.from_csv()
DEMO = dict(age=21, height_cm=175, weight_kg=70, fitness_level="Intermediate", goal="Muscle Gain",
            target_muscle="Chest", equipment="Full Gym", duration=60, days_per_week=5,
            exercise_type="Strength")


def make(**overrides) -> UserProfile:
    return UserProfile(**{**DEMO, **overrides})


def sample_profiles() -> list[tuple[str, UserProfile]]:
    out = []
    for d in json.loads((ROOT / "data" / "sample_users.json").read_text(encoding="utf-8")):
        d = dict(d)
        out.append((d.pop("label"), UserProfile.from_dict(d)))
    return out


# --------------------------------------------------------------------- dataset
def test_dataset_schema_and_size():
    raw = load_raw_dataset()
    assert len(raw) >= 250
    assert set(REQUIRED_COLUMNS).issubset(raw.columns)
    assert raw["exercise_id"].is_unique and raw["exercise_name"].is_unique
    assert len(REC.df) == len(raw)


def test_cleaning_handles_bad_rows():
    raw = load_raw_dataset()
    dirty = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    dirty.loc[1, "muscle_group"] = None
    dirty.loc[2, "duration_minutes"] = -4
    dirty["calories_burned"] = dirty["calories_burned"].astype(object)
    dirty.loc[3, "calories_burned"] = "abc"
    clean, report = clean_exercise_data(dirty)
    assert report.duplicates_removed == 1
    assert report.unknown_categories_dropped == 1
    assert sum(report.invalid_numeric_fixed.values()) >= 2
    assert clean[REQUIRED_COLUMNS].isna().sum().sum() == 0
    assert (clean["duration_minutes"] > 0).all()


# --------------------------------------------------------------------- profile
def test_profile_validation():
    _, errors = build_profile({**DEMO, "age": 5, "height_cm": "abc", "goal": None})
    assert any("Age" in e for e in errors)
    assert any("Height" in e for e in errors)
    assert any("goal" in e.lower() for e in errors)
    profile, errors = build_profile(DEMO)
    assert errors == [] and profile is not None
    assert profile.bmi == 22.9


# ---------------------------------------------------------------- recommender
def test_cosine_matches_manual_formula():
    profile = make()
    u = REC.encoder.encode_user(profile).ravel()
    result = REC.recommend(profile, top_n=10)
    for _, row in result.top_n.iterrows():
        e = REC.X[int(row["row_index"])]
        manual = u @ e / (np.linalg.norm(u) * np.linalg.norm(e))
        assert np.isclose(manual, row["similarity"])
        contrib = sum(row[f"contrib_{b}"] for b in BLOCK_ORDER)
        assert np.isclose(contrib, row["similarity"])


def test_scores_bounded_and_sorted():
    result = REC.recommend(make(), top_n=10)
    s = result.ranked["similarity"].to_numpy()
    assert (s >= 0).all() and (s <= 1 + 1e-9).all()
    assert (np.diff(s) <= 1e-12).all()
    assert list(result.top_n["rank"]) == list(range(1, 11))


def test_demo_user_recommends_chest_gym_exercises():
    top = REC.recommend(make(), top_n=5).top_n
    assert (top["muscle_group"] == "Chest").all()
    assert set(top["difficulty"]) <= {"Beginner", "Intermediate"}


def test_no_equipment_user_never_gets_equipment():
    for goal in GOALS:
        for target in TARGET_MUSCLES:
            p = make(equipment="No Equipment", goal=goal, target_muscle=target,
                     fitness_level="Beginner", exercise_type="Mixed")
            result = REC.recommend(p, top_n=10)
            assert not result.ranked.empty
            assert set(result.ranked["equipment"]) <= set(EQUIPMENT_ACCESS["No Equipment"])


def test_beginner_strict_stage_has_no_advanced():
    result = REC.recommend(make(fitness_level="Beginner"), top_n=5)
    if result.filter_stage_index == 0:
        assert "Advanced" not in set(result.ranked["difficulty"])
    assert "Advanced" not in set(result.top_n["difficulty"])


def test_every_profile_combination_returns_results():
    """Never-empty guarantee across all categorical combinations."""
    for level in FITNESS_LEVELS:
        for goal in GOALS:
            for target in TARGET_MUSCLES:
                for equip in USER_EQUIPMENT:
                    for etype in USER_EXERCISE_TYPES:
                        p = make(fitness_level=level, goal=goal, target_muscle=target,
                                 equipment=equip, exercise_type=etype)
                        rows, scores = REC.rank_indices(p, 5)
                        assert len(rows) >= 1
                        assert set(REC.df["equipment"].to_numpy()[rows]) <= set(EQUIPMENT_ACCESS[equip])


def test_explanations_are_dynamic():
    profile = make()
    top = REC.recommend(profile, top_n=3).top_n
    reasons = [str(r) for r in explain_exercise(top.iloc[0], profile)]
    assert any("Muscle Gain" in r for r in reasons)
    assert any("Chest" in r for r in reasons)
    other = make(goal="Strength", target_muscle="Legs")
    top2 = REC.recommend(other, top_n=1).top_n
    assert [str(r) for r in explain_exercise(top2.iloc[0], other)] != reasons


# ----------------------------------------------------------- workout generator
def test_workout_counts_and_duration():
    for duration in DURATIONS:
        p = make(duration=duration)
        _, w = recommend_and_generate(REC, p)
        lo, hi = EXERCISE_COUNT_RANGE[duration]
        assert lo <= len(w.items) <= hi, (duration, len(w.items))
        assert w.total_minutes <= duration * 1.05 + 0.5, (duration, w.total_minutes)
        assert len(set(w.exercise_ids)) == len(w.items)


def test_sample_users_all_work():
    for label, p in sample_profiles():
        result, w = recommend_and_generate(REC, p, top_n=5)
        assert len(result.top_n) == 5, label
        assert w.items, label
        assert w.total_calories(p.weight_kg) > 0


def test_regenerate_gives_different_workout():
    result, w = recommend_and_generate(REC, make())
    alt = generate_workout(result, REC, variation=1, exclude_ids=w.exercise_ids)
    assert set(alt.exercise_ids) != set(w.exercise_ids)
    assert set(alt.exercise_ids) <= set(result.ranked["exercise_id"])   # same constraints


def test_deterministic_first_workout():
    _, a = recommend_and_generate(REC, make())
    _, b = recommend_and_generate(REC, make())
    assert a.exercise_ids == b.exercise_ids


# --------------------------------------------------------------------- storage
def test_storage_recovers_from_corruption():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "history.json"
        path.write_text("{not valid json", encoding="utf-8")
        store = JSONListStore(path)
        assert store.load() == []
        assert store.last_warning is not None
        store.append({"goal": "Strength"})
        assert len(store.load()) == 1
        assert any(p.name.startswith("history.corrupt-") for p in Path(tmp).iterdir())


def test_shipped_json_files_valid():
    for name in ("workout_history.json", "feedback.json", "sample_users.json"):
        data = json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))
        assert isinstance(data, list)


# ------------------------------------------------------------------ runner
if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {exc!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
