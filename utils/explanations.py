"""
"Why this exercise?" explanations.

Every explanation is generated from two things only:
  1. the user's actual selections (UserProfile), and
  2. the exercise's attributes + its per-block cosine similarities computed
     by the recommender (columns sim_goal, sim_muscle, ...).

Nothing is hard-coded per exercise. Each reason carries a status:
  "match"   - the exercise matches this preference         (shown as ✓)
  "partial" - related but not an exact match               (shown as ◐)
  "note"    - a trade-off the user should know about       (shown as ⚠)
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from model.config import (BLOCK_LABELS, CONDITIONING_GOALS, EQUIPMENT_ACCESS,
                          FITNESS_LEVELS, MUSCLE_AFFINITY)
from model.recommender import BLOCK_ORDER, per_exercise_time_budget
from model.user_profile import UserProfile

ICONS = {"match": "✓", "partial": "◐", "note": "⚠"}


@dataclass
class Reason:
    status: str
    text: str

    @property
    def icon(self) -> str:
        return ICONS[self.status]

    def __str__(self) -> str:
        return f"{self.icon} {self.text}"


def explain_exercise(row: pd.Series, profile: UserProfile) -> list[Reason]:
    """Build the list of reasons for one recommended exercise."""
    reasons: list[Reason] = []
    goals = list(row["goals_list"]) if "goals_list" in row else str(row["goal"]).split("|")

    # --- goal -----------------------------------------------------------------
    if goals and goals[0] == profile.goal:
        reasons.append(Reason("match", f"Matches your {profile.goal} goal (its primary purpose)"))
    elif profile.goal in goals:
        reasons.append(Reason("match", f"Supports your {profile.goal} goal "
                                       f"(mainly a {goals[0]} exercise)"))
    else:
        reasons.append(Reason("partial", f"Designed for {', '.join(goals)}, which is related to "
                                         f"your {profile.goal} goal"))

    # --- target muscle --------------------------------------------------------
    muscle = row["muscle_group"]
    if muscle == profile.target_muscle:
        reasons.append(Reason("match", f"Targets your selected {profile.target_muscle} muscle group"))
    elif muscle in ("Full Body", "Cardio") and profile.goal in CONDITIONING_GOALS:
        reasons.append(Reason("match", f"{muscle} conditioning that supports {profile.goal}"))
    elif MUSCLE_AFFINITY[profile.target_muscle].get(muscle, 0) > 0:
        reasons.append(Reason("partial", f"Works {muscle}, which assists {profile.target_muscle} training"))
    else:
        reasons.append(Reason("note", f"Works {muscle} (added because few {profile.target_muscle} "
                                      f"options fit all your settings)"))

    # --- fitness level ----------------------------------------------------------
    diff = row["difficulty"]
    gap = FITNESS_LEVELS.index(diff) - FITNESS_LEVELS.index(profile.fitness_level)
    if gap == 0:
        reasons.append(Reason("match", f"Suitable for {profile.fitness_level} level"))
    elif gap < 0:
        reasons.append(Reason("partial", f"{diff} difficulty - comfortable for your "
                                         f"{profile.fitness_level} level"))
    else:
        reasons.append(Reason("note", f"{diff} difficulty - a step above your level; "
                                      f"start light and focus on form"))

    # --- equipment --------------------------------------------------------------
    equip = row["equipment"]
    if equip in EQUIPMENT_ACCESS[profile.equipment]:
        if equip == "No Equipment":
            reasons.append(Reason("match", "Needs no equipment"))
        else:
            reasons.append(Reason("match", f"Uses {equip.lower()}, available with your "
                                           f"{profile.equipment} setup"))

    # --- exercise type ----------------------------------------------------------
    etype = row["exercise_type"]
    if profile.exercise_type == "Mixed":
        reasons.append(Reason("match", f"Adds {etype} work to your mixed-training preference"))
    elif etype == profile.exercise_type:
        reasons.append(Reason("match", f"Matches your preferred {etype} training"))
    elif row.get("sim_type", 0) >= 0.5:
        reasons.append(Reason("partial", f"{etype} exercise - complements your {profile.exercise_type} "
                                         f"preference and {profile.goal} goal"))
    else:
        reasons.append(Reason("note", f"{etype} exercise (you preferred {profile.exercise_type}); "
                                      f"chosen for its strong match on other factors"))

    # --- duration ---------------------------------------------------------------
    budget = per_exercise_time_budget(profile.duration)
    minutes = float(row["duration_minutes"])
    if row.get("sim_duration", 0) >= 0.6:
        reasons.append(Reason("match", f"Fits your {profile.duration}-minute session "
                                       f"(about {minutes:g} min of ~{budget:.0f} min per exercise)"))
    else:
        reasons.append(Reason("partial", f"Takes about {minutes:g} min; sets are adjusted to fit "
                                         f"your {profile.duration}-minute session"))

    # --- keyword profile (TF-IDF) -------------------------------------------------
    if row.get("sim_text", 0) >= 0.3:
        reasons.append(Reason("match", "Movement description closely matches your preferences "
                                       "(TF-IDF keyword similarity)"))
    return reasons


def score_breakdown(row: pd.Series) -> pd.DataFrame:
    """
    How much each feature block contributed to the final cosine score.
    The contributions add up exactly to the match score.
    """
    records = []
    for b in BLOCK_ORDER:
        records.append({
            "Feature": BLOCK_LABELS[b],
            "Block similarity": float(row.get(f"sim_{b}", 0.0)),
            "Contribution (%)": round(100 * float(row.get(f"contrib_{b}", 0.0)), 1),
        })
    return pd.DataFrame(records)


def summary_sentence(row: pd.Series, profile: UserProfile) -> str:
    """One-line summary used in compact cards."""
    matches = sum(r.status == "match" for r in explain_exercise(row, profile))
    return f"{row['match_percent']:.0f}% match - fits {matches} of your preferences"
