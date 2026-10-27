"""
Evaluation of the content-based recommender
===========================================

  *** DEMONSTRATION / TEST EVALUATION - NOT REAL-WORLD PERFORMANCE ***

Genuine Precision@K and Recall@K require RELEVANCE LABELS: for each user, a
list of exercises that a real person (or a qualified trainer) judged
relevant. This project has no real users and no expert labels yet.

To still demonstrate HOW the metrics are calculated, this module builds a
clearly labelled SYNTHETIC relevance set with transparent rules (a "proxy
oracle"): an exercise is treated as relevant for a profile when

  * its equipment is available to the user, AND
  * it lists the user's goal among its goals, AND
  * it trains the target muscle (or, for Weight Loss / Endurance, it is a
    full-body / cardio conditioning exercise), AND
  * its difficulty suits the user's level, AND
  * it is not a pure mobility drill (unless the user prefers Mobility or
    the goal is General Fitness).

Because these rules use the same exercise attributes the recommender sees,
the numbers are OPTIMISTIC and only show that the pipeline works as
designed. They must not be presented as real-world accuracy.

Metrics
-------
  Precision@K = |recommended_K  ∩ relevant| / K
  Recall@K    = |recommended_K  ∩ relevant| / |relevant|
  nDCG@K      = DCG@K / ideal DCG@K   (binary relevance, rank-aware)
  Cosine      = (A · B) / (||A|| × ||B||) - verified manually vs scikit-learn

Baselines (to show the value of each pipeline stage)
  * Random            - K random exercises, no personalisation
  * Filter only       - K random exercises from the hard-filtered pool
  * Content-based     - the full system: filter + cosine ranking (ours)

Run from the project root:

    python evaluation/evaluate.py              # full grid (4,860 profiles)
    python evaluation/evaluate.py --quick      # 600 sampled profiles
    python evaluation/evaluate.py --write-labels   # regenerate label file
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.config import (CONDITIONING_GOALS, EQUIPMENT_ACCESS, FITNESS_LEVELS,  # noqa: E402
                          GOALS, TARGET_MUSCLES, USER_EQUIPMENT,
                          USER_EXERCISE_TYPES)
from model.recommender import WorkoutRecommender  # noqa: E402
from model.user_profile import UserProfile  # noqa: E402

SAMPLE_USERS_PATH = ROOT / "data" / "sample_users.json"
LABELS_PATH = ROOT / "evaluation" / "test_relevance_labels.csv"
RESULTS_DIR = ROOT / "evaluation" / "results"

PROXY_LEVEL_OK = {"Beginner": {"Beginner"},
                  "Intermediate": {"Beginner", "Intermediate"},
                  "Advanced": {"Intermediate", "Advanced"}}


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-K recommendations that are relevant."""
    if k <= 0:
        return 0.0
    top = recommended[:k]
    return sum(1 for r in top if r in relevant) / k


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Fraction of all relevant items that appear in the top-K."""
    if not relevant:
        return float("nan")  # undefined without relevant items
    top = recommended[:k]
    return sum(1 for r in top if r in relevant) / len(relevant)


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance."""
    gains = [1.0 if r in relevant else 0.0 for r in recommended[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant))))
    return dcg / ideal if ideal > 0 else float("nan")


def cosine_similarity_manual(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity written out exactly as in the report: (A·B)/(||A||·||B||)."""
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


# ----------------------------------------------------------------------------
# Synthetic relevance labels (proxy oracle)
# ----------------------------------------------------------------------------
def proxy_relevant_ids(df: pd.DataFrame, profile: UserProfile) -> set[str]:
    """Rule-based SYNTHETIC relevance labels for one profile (see module docstring)."""
    equip_ok = df["equipment"].isin(EQUIPMENT_ACCESS[profile.equipment])
    goal_ok = df["goals_list"].map(lambda g: profile.goal in g)
    if profile.target_muscle == "Full Body":
        muscle_ok = df["muscle_group"].eq("Full Body")
    else:
        muscle_ok = df["muscle_group"].eq(profile.target_muscle)
    if profile.goal in CONDITIONING_GOALS:
        muscle_ok |= df["muscle_group"].isin(["Full Body", "Cardio"]) & \
            df["exercise_type"].isin(["HIIT", "Cardio"])
    level_ok = df["difficulty"].isin(PROXY_LEVEL_OK[profile.fitness_level])
    type_ok = df["exercise_type"].ne("Mobility") | (profile.exercise_type == "Mobility") | \
        (profile.goal == "General Fitness")
    mask = equip_ok & goal_ok & muscle_ok & level_ok & type_ok
    return set(df.loc[mask, "exercise_id"])


def load_sample_users(path: Path = SAMPLE_USERS_PATH) -> list[tuple[str, UserProfile]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [(d["label"], UserProfile.from_dict(d)) for d in data]


def write_label_file(rec: WorkoutRecommender, path: Path = LABELS_PATH) -> pd.DataFrame:
    """Store the synthetic labels for the 5 sample users (for transparency)."""
    rows = []
    for label, profile in load_sample_users():
        for ex_id in sorted(proxy_relevant_ids(rec.df, profile)):
            rows.append({"profile": label, "exercise_id": ex_id, "relevant": 1,
                         "label_source": "SYNTHETIC proxy rules (demonstration only)"})
    out = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def load_label_file(path: Path = LABELS_PATH) -> dict[str, set[str]]:
    if not Path(path).exists():
        return {}
    labels = pd.read_csv(path)
    return {p: set(g["exercise_id"]) for p, g in labels.groupby("profile")}


# ----------------------------------------------------------------------------
# Evaluation runs
# ----------------------------------------------------------------------------
def _ids(rec: WorkoutRecommender, rows: np.ndarray) -> list[str]:
    return rec.df["exercise_id"].to_numpy()[rows].tolist()


def evaluate_profile(rec: WorkoutRecommender, profile: UserProfile, relevant: set[str],
                     k: int, rng: np.random.Generator, random_trials: int = 5) -> dict:
    """Metrics for one profile: ours vs. two baselines."""
    rows, scores = rec.rank_indices(profile, k)
    ours = _ids(rec, rows)

    mask, _ = rec.filter_candidates(profile, k)
    eligible = np.flatnonzero(mask)
    all_rows = np.arange(len(rec.df))

    def rand_metrics(pool: np.ndarray) -> tuple[float, float, float]:
        p, r, n = [], [], []
        for _ in range(random_trials):
            pick = _ids(rec, rng.choice(pool, size=min(k, len(pool)), replace=False))
            p.append(precision_at_k(pick, relevant, k))
            r.append(recall_at_k(pick, relevant, k))
            n.append(ndcg_at_k(pick, relevant, k))
        return float(np.mean(p)), float(np.mean(r)), float(np.mean(n))

    rp, rr, rn = rand_metrics(all_rows)
    fp, fr, fn = rand_metrics(eligible)
    return {
        "n_relevant": len(relevant),
        "ours_precision": precision_at_k(ours, relevant, k),
        "ours_recall": recall_at_k(ours, relevant, k),
        "ours_ndcg": ndcg_at_k(ours, relevant, k),
        "ours_mean_cosine": float(np.mean(scores)) if len(scores) else float("nan"),
        "filter_precision": fp, "filter_recall": fr, "filter_ndcg": fn,
        "random_precision": rp, "random_recall": rr, "random_ndcg": rn,
    }


def profile_grid(max_profiles: int | None = None, seed: int = 7) -> list[UserProfile]:
    """All combinations of level x goal x target x equipment x type (4,860)."""
    combos = list(itertools.product(FITNESS_LEVELS, GOALS, TARGET_MUSCLES,
                                    USER_EQUIPMENT, USER_EXERCISE_TYPES))
    if max_profiles is not None and max_profiles < len(combos):
        idx = np.random.default_rng(seed).choice(len(combos), size=max_profiles, replace=False)
        combos = [combos[i] for i in sorted(idx)]
    return [UserProfile(age=25, height_cm=172, weight_kg=70, fitness_level=lvl, goal=g,
                        target_muscle=t, equipment=e, duration=60, days_per_week=4,
                        exercise_type=et)
            for lvl, g, t, e, et in combos]


def evaluate_sample_users(rec: WorkoutRecommender, k: int = 5, seed: int = 0) -> pd.DataFrame:
    """Per-user table for the 5 sample profiles using the stored label file."""
    labels = load_label_file()
    rng = np.random.default_rng(seed)
    rows = []
    for label, profile in load_sample_users():
        relevant = labels.get(label) or proxy_relevant_ids(rec.df, profile)
        m = evaluate_profile(rec, profile, relevant, k, rng)
        rows.append({"Profile": label, **m})
    return pd.DataFrame(rows)


def evaluate_grid(rec: WorkoutRecommender, ks: tuple[int, ...] = (5, 10),
                  max_profiles: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Mean metrics over the profile grid for each K (profiles with no relevant items skipped)."""
    rng = np.random.default_rng(seed)
    profiles = profile_grid(max_profiles)
    labelled = [(p, proxy_relevant_ids(rec.df, p)) for p in profiles]
    usable = [(p, rel) for p, rel in labelled if rel]
    out = []
    for k in ks:
        res = pd.DataFrame([evaluate_profile(rec, p, rel, k, rng, random_trials=2)
                            for p, rel in usable])
        for system, prefix in (("Content-based cosine (ours)", "ours"),
                               ("Filter only (no ranking)", "filter"),
                               ("Random (no personalisation)", "random")):
            out.append({"K": k, "System": system,
                        f"Precision@K": res[f"{prefix}_precision"].mean(),
                        f"Recall@K": res[f"{prefix}_recall"].mean(),
                        f"nDCG@K": res[f"{prefix}_ndcg"].mean(),
                        "Profiles evaluated": len(usable),
                        "Profiles skipped (no relevant items)": len(labelled) - len(usable)})
    return pd.DataFrame(out)


def cosine_verification(rec: WorkoutRecommender) -> pd.DataFrame:
    """Show that the manual formula equals scikit-learn for the demo user."""
    _, profile = load_sample_users()[1]
    user_vec = rec.encoder.encode_user(profile)
    rows, sk_scores = rec.rank_indices(profile, 5)
    records = []
    for r, s in zip(rows, sk_scores):
        manual = cosine_similarity_manual(user_vec, rec.X[r])
        records.append({"Exercise": rec.df["exercise_name"].iloc[r],
                        "A·B": float(user_vec.ravel() @ rec.X[r]),
                        "||A||": float(np.linalg.norm(user_vec)),
                        "||B||": float(np.linalg.norm(rec.X[r])),
                        "Manual cosine": manual, "scikit-learn cosine": float(s),
                        "Equal": bool(np.isclose(manual, s))})
    return pd.DataFrame(records)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstration evaluation of the recommender")
    parser.add_argument("--quick", action="store_true", help="evaluate 600 sampled profiles")
    parser.add_argument("--write-labels", action="store_true",
                        help="regenerate evaluation/test_relevance_labels.csv")
    args = parser.parse_args()

    rec = WorkoutRecommender.from_csv()
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("=" * 78)
    print(" DEMONSTRATION / TEST EVALUATION - synthetic relevance labels")
    print(" These numbers show how the metrics work; they are NOT real-world accuracy.")
    print("=" * 78)

    if args.write_labels or not LABELS_PATH.exists():
        labels = write_label_file(rec)
        print(f"\nWrote {len(labels)} synthetic labels to {LABELS_PATH.relative_to(ROOT)}")

    print("\n[1] Cosine similarity check (manual formula vs scikit-learn), demo user:")
    print(cosine_verification(rec).round(4).to_string(index=False))

    print("\n[2] Sample users (K = 5):")
    su = evaluate_sample_users(rec, k=5)
    cols = ["Profile", "n_relevant", "ours_precision", "ours_recall", "ours_ndcg",
            "filter_precision", "random_precision"]
    print(su[cols].round(3).to_string(index=False))

    n = 600 if args.quick else None
    print(f"\n[3] Profile grid ({'600 sampled' if n else 'all 4,860'} profiles):")
    grid = evaluate_grid(rec, max_profiles=n)
    print(grid.round(3).to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    su.to_csv(RESULTS_DIR / "demo_sample_user_metrics.csv", index=False)
    grid.to_csv(RESULTS_DIR / "demo_grid_metrics.csv", index=False)
    print(f"\nSaved CSV results to {RESULTS_DIR.relative_to(ROOT)}/ (labelled as demonstration).")


if __name__ == "__main__":
    main()
