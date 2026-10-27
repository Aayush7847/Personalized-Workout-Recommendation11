"""
Personalized workout generator.

Turns the ranked recommendation list into a complete, time-boxed session.

Steps
-----
1. Candidate pool  : the eligible exercises ranked by cosine similarity.
2. Selection       : Maximal Marginal Relevance (MMR) - balances a high match
                     score against similarity to exercises already chosen, so
                     a chest workout is not five near-identical presses.
                         MMR(i) = lambda * score(i) - (1 - lambda) * max_j sim(i, j)
                     "Generate another workout" adds small random noise to the
                     scores (seeded per regeneration) and penalises the
                     previous workout's exercises, so new ones are preferred
                     whenever good alternatives exist.
3. Prescription    : sets / reps / rest from the dataset, adapted to the user's
                     goal (strength = low reps, long rest; endurance = high reps,
                     short rest, ...) and level.
4. Time fitting    : number of exercises follows the session length
                     (20-30 min: 3-4, 45: 4-5, 60: 5-6, 90: 6-8); sets are then
                     added / removed so the estimate lands close to the target.
5. Ordering        : compound strength -> isolation -> HIIT -> cardio -> core
                     -> mobility (a conventional session structure).

All prescriptions are educational suggestions, not professional advice.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from model.config import (COOLDOWN_MINUTES, EXERCISE_COUNT_RANGE, GOAL_PRESCRIPTION,
                          MAX_SETS, MIN_SETS, REFERENCE_BODY_WEIGHT_KG,
                          WARMUP_KCAL_PER_MIN, WARMUP_MINUTES)
from model.recommender import RecommendationResult, WorkoutRecommender
from model.user_profile import UserProfile
from utils.timing import estimate_exercise_seconds, is_timed

MMR_LAMBDA = 0.72            # 1.0 = pure relevance, 0.0 = pure diversity
REGEN_NOISE = 0.035          # Gumbel noise scale used for "Generate another"
REPEAT_PENALTY = 0.10        # score penalty for exercises in the previous workout
QUALITY_FLOOR = 0.80         # candidates must score >= 80% of the best score
REGEN_QUALITY_FLOOR = 0.70   # slightly wider pool when generating an alternative workout
POOL_FACTOR = 2              # eligible pool requested from the filter = 2 x max exercises
MAX_PER_FAMILY = 2           # at most 2 exercises from the same movement family
POOL_MULTIPLIER = 4          # candidate pool = 4 x max exercises
MAX_SETS_BY_LEVEL = {"Beginner": 4, "Intermediate": MAX_SETS, "Advanced": MAX_SETS}

# Movement patterns that train the same thing are grouped into families so a
# workout is not built from near-identical movements.
PATTERN_FAMILY = {"h_push": "horizontal_press", "incline_push": "horizontal_press",
                  "v_pull": "vertical_pull", "hang": "vertical_pull",
                  "plank": "core_hold", "iso_hold": "core_hold",
                  "cardio_steady": "steady_cardio"}
UPPER_TOLERANCE = 1.05       # allow up to +5% of the requested duration
LOWER_TARGET = 0.90          # try to reach at least 90% of the requested duration

PHASE_ORDER = {"Strength-Compound": 0, "Strength-Isolation": 1, "HIIT": 2,
               "Cardio": 3, "Core": 4, "Mobility": 5}


@dataclass
class WorkoutItem:
    order: int
    exercise_id: str
    exercise_name: str
    muscle_group: str
    equipment: str
    difficulty: str
    exercise_type: str
    sets: int
    reps: str
    rest_seconds: int
    est_minutes: float
    calories: float
    match_percent: float
    instructions: str
    safety_notes: str
    row_index: int


@dataclass
class Workout:
    goal: str
    target_muscle: str
    fitness_level: str
    equipment: str
    requested_minutes: int
    warmup_minutes: int
    cooldown_minutes: int
    items: list[WorkoutItem] = field(default_factory=list)
    variation: int = 0

    @property
    def exercise_minutes(self) -> float:
        return round(sum(i.est_minutes for i in self.items), 1)

    @property
    def total_minutes(self) -> float:
        return round(self.exercise_minutes + self.warmup_minutes + self.cooldown_minutes, 1)

    def total_calories(self, weight_kg: float) -> int:
        scale = weight_kg / REFERENCE_BODY_WEIGHT_KG
        warm = (self.warmup_minutes + self.cooldown_minutes) * WARMUP_KCAL_PER_MIN * scale
        return int(round(sum(i.calories for i in self.items) + warm))

    @property
    def exercise_ids(self) -> list[str]:
        return [i.exercise_id for i in self.items]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(i) for i in self.items])


# ----------------------------------------------------------------------------
# Prescription
# ----------------------------------------------------------------------------
def _phase(row: pd.Series) -> str:
    t = row["exercise_type"]
    if t == "Strength":
        return "Strength-Compound" if row["mechanics"] == "Compound" else "Strength-Isolation"
    return t


def prescribe(row: pd.Series, profile: UserProfile) -> tuple[int, str, int]:
    """
    Sets / reps / rest for one exercise.

    * Timed exercises (cardio, HIIT, holds, mobility) keep the dataset's
      time-based prescription.
    * Core exercises keep the dataset's core prescription.
    * Repetition-based strength exercises use the dataset values when the
      exercise's primary goal equals the user's goal; otherwise the goal
      template (config.GOAL_PRESCRIPTION) is applied. Beginners get one set
      fewer (minimum 2).
    """
    sets, reps, rest = int(row["sets"]), str(row["reps"]), int(row["rest_seconds"])
    if is_timed(reps) or row["exercise_type"] in ("Core", "Mobility", "Cardio", "HIIT"):
        return sets, reps, rest
    if row["primary_goal"] == profile.goal:
        return sets, reps, rest
    kind = "compound" if row["mechanics"] == "Compound" else "isolation"
    sets, reps, rest = GOAL_PRESCRIPTION[profile.goal][kind]
    if profile.fitness_level == "Beginner":
        sets = max(MIN_SETS, sets - 1)
    return sets, reps, rest


def _steady_minutes(reps: str) -> int | None:
    """'15 min' -> 15 for single-set steady cardio, else None."""
    m = re.search(r"(\d+)\s*min", reps)
    return int(m.group(1)) if m else None


# ----------------------------------------------------------------------------
# Selection (MMR + optional variation)
# ----------------------------------------------------------------------------
def _select_mmr(ranked: pd.DataFrame, recommender: WorkoutRecommender, k: int,
                rng: np.random.Generator | None, previous: set[str],
                floor_ratio: float = QUALITY_FLOOR) -> pd.DataFrame:
    """
    Pick k diverse, high-scoring exercises with Maximal Marginal Relevance.

    * Quality floor: only candidates scoring >= floor_ratio x best score are
      considered first; lower-scoring ones are used only if the workout would
      otherwise be short or too repetitive. Alternative workouts use a slightly lower floor so that genuinely
      different, still well-matched exercises can be chosen.
    * Regeneration: exercises from the previous workout get a soft penalty
      (they are only repeated when alternatives are clearly worse), and small
      seeded Gumbel noise reshuffles near-ties.
    * Movement-family cap: at most MAX_PER_FAMILY per family (e.g. two squat
      variations); dropped only if the eligible pool is too small.
    """
    if ranked.empty:
        return ranked
    pool = ranked.head(max(k * POOL_MULTIPLIER, 12))
    floor = floor_ratio * float(pool["similarity"].iloc[0])
    above_floor = pool["similarity"].to_numpy() >= floor

    relevance = pool["similarity"].to_numpy().copy()
    if previous:
        relevance -= REPEAT_PENALTY * pool["exercise_id"].isin(previous).to_numpy()
    if rng is not None:
        relevance += REGEN_NOISE * rng.gumbel(size=len(relevance))

    rows = pool["row_index"].to_numpy()
    sim = recommender.exercise_similarity(rows, rows)
    families = pool["movement_pattern"].map(lambda p: PATTERN_FAMILY.get(p, p)).to_numpy()

    def run(family_cap: int | None, allowed: np.ndarray, seed: list[int]) -> list[int]:
        """Greedy MMR: relevance minus redundancy with what is already chosen."""
        chosen = list(seed)
        count: dict[str, int] = {}
        for c in chosen:
            count[families[c]] = count.get(families[c], 0) + 1
        available = allowed.copy()
        available[chosen] = False
        while len(chosen) < k and available.any():
            redundancy = sim[:, chosen].max(axis=1) if chosen else np.zeros(len(pool))
            mmr = MMR_LAMBDA * relevance - (1 - MMR_LAMBDA) * redundancy
            mmr[~available] = -np.inf
            best = int(np.argmax(mmr))
            available[best] = False
            if family_cap is not None and count.get(families[best], 0) >= family_cap:
                continue
            chosen.append(best)
            count[families[best]] = count.get(families[best], 0) + 1
        return chosen

    everything = np.ones(len(pool), dtype=bool)
    # 1) well-matched candidates, movement-family cap respected
    chosen = run(MAX_PER_FAMILY, above_floor, [])
    # 2) still short -> look below the quality floor, keeping the family cap
    if len(chosen) < k:
        chosen = run(MAX_PER_FAMILY, everything, chosen)
    # 3) still short (tiny pool) -> drop the family cap
    if len(chosen) < k:
        chosen = run(None, everything, chosen)
    return pool.iloc[chosen]


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def generate_workout(result: RecommendationResult, recommender: WorkoutRecommender,
                     variation: int = 0, exclude_ids: list[str] | None = None) -> Workout:
    """
    Build a complete workout from a recommendation result.

    variation = 0  -> best workout (deterministic)
    variation > 0  -> an alternative workout (seeded randomness; exercises in
                      `exclude_ids` are penalised so different ones are
                      preferred whenever good alternatives exist)
    """
    profile = result.profile
    duration = int(profile.duration)
    n_min, n_max = EXERCISE_COUNT_RANGE.get(duration, (4, 5))
    warm, cool = WARMUP_MINUTES.get(duration, 5), COOLDOWN_MINUTES.get(duration, 3)
    target_sec = duration * 60.0
    fixed_sec = (warm + cool) * 60.0

    rng = np.random.default_rng(variation) if variation > 0 else None
    floor_ratio = REGEN_QUALITY_FLOOR if variation > 0 else QUALITY_FLOOR
    selected = _select_mmr(result.ranked, recommender, n_max, rng, set(exclude_ids or []),
                           floor_ratio)
    max_sets = MAX_SETS_BY_LEVEL.get(profile.fitness_level, MAX_SETS)

    # --- prescriptions + time per exercise --------------------------------------
    plan = []
    for _, row in selected.iterrows():
        sets, reps, rest = prescribe(row, profile)
        plan.append({"row": row, "sets": sets, "reps": reps, "rest": rest,
                     "added": 0, "score": float(row["similarity"])})

    def ex_seconds(p: dict) -> float:
        return estimate_exercise_seconds(p["sets"], p["reps"], p["rest"])

    # --- choose how many exercises (within the range for this duration) ---------
    n = min(n_min, len(plan))
    total = fixed_sec + sum(ex_seconds(p) for p in plan[:n])
    while n < min(n_max, len(plan)) and total + ex_seconds(plan[n]) <= target_sec * UPPER_TOLERANCE:
        total += ex_seconds(plan[n])
        n += 1
    plan = plan[:n]

    def total_seconds() -> float:
        return fixed_sec + sum(ex_seconds(p) for p in plan)

    # --- fill: add sets (or cardio minutes) until close to the target -----------
    while total_seconds() < target_sec * LOWER_TARGET:
        candidates = []
        for p in plan:
            steady = _steady_minutes(p["reps"])
            if steady is not None and p["sets"] == 1:
                if steady < 30:
                    candidates.append(p)
            elif p["sets"] < max_sets:
                candidates.append(p)
        if not candidates:
            break
        p = min(candidates, key=lambda q: (q["added"], -q["score"]))
        steady = _steady_minutes(p["reps"])
        if steady is not None and p["sets"] == 1:
            p["reps"] = f"{steady + 5} min"
        else:
            p["sets"] += 1
        p["added"] += 1
        if total_seconds() > target_sec * UPPER_TOLERANCE:  # overshoot -> undo
            if steady is not None and p["sets"] == 1:
                p["reps"] = f"{steady} min"
            else:
                p["sets"] -= 1
            break

    # --- trim: remove sets, then exercises, if over the target ------------------
    while total_seconds() > target_sec * UPPER_TOLERANCE:
        reducible = [p for p in plan if p["sets"] > MIN_SETS or
                     ((_steady_minutes(p["reps"]) or 0) > 5 and p["sets"] == 1)]
        if reducible:
            p = min(reducible, key=lambda q: q["score"])
            steady = _steady_minutes(p["reps"])
            if steady is not None and p["sets"] == 1:
                p["reps"] = f"{max(5, steady - 5)} min"
            else:
                p["sets"] -= 1
        elif len(plan) > n_min:
            plan.remove(min(plan, key=lambda q: q["score"]))
        else:
            break

    # --- order and package --------------------------------------------------------
    plan.sort(key=lambda p: (PHASE_ORDER.get(_phase(p["row"]), 9), -p["score"]))
    scale = profile.weight_kg / REFERENCE_BODY_WEIGHT_KG
    items = []
    for i, p in enumerate(plan, start=1):
        row = p["row"]
        minutes = round(ex_seconds(p) / 60.0, 1)
        items.append(WorkoutItem(
            order=i, exercise_id=str(row["exercise_id"]), exercise_name=str(row["exercise_name"]),
            muscle_group=str(row["muscle_group"]), equipment=str(row["equipment"]),
            difficulty=str(row["difficulty"]), exercise_type=str(row["exercise_type"]),
            sets=int(p["sets"]), reps=str(p["reps"]), rest_seconds=int(p["rest"]),
            est_minutes=minutes, calories=round(float(row["kcal_per_min"]) * minutes * scale, 1),
            match_percent=float(row["match_percent"]), instructions=str(row["instructions"]),
            safety_notes=str(row["safety_notes"]), row_index=int(row["row_index"]),
        ))

    return Workout(goal=profile.goal, target_muscle=profile.target_muscle,
                   fitness_level=profile.fitness_level, equipment=profile.equipment,
                   requested_minutes=duration, warmup_minutes=warm, cooldown_minutes=cool,
                   items=items, variation=variation)


def recommend_and_generate(recommender: WorkoutRecommender, profile: UserProfile,
                           top_n: int = 5, variation: int = 0,
                           exclude_ids: list[str] | None = None
                           ) -> tuple[RecommendationResult, Workout]:
    """Convenience wrapper used by the app, notebook and tests."""
    n_max = EXERCISE_COUNT_RANGE.get(int(profile.duration), (4, 5))[1]
    # Ask for an eligible pool big enough to build a full, diverse workout and
    # to offer alternatives on "Generate Another Workout" (the filter relaxes
    # step by step only if the strict pool is smaller than this).
    result = recommender.recommend(profile, top_n=top_n, min_candidates=POOL_FACTOR * n_max)
    workout = generate_workout(result, recommender, variation=variation, exclude_ids=exclude_ids)
    return result, workout
