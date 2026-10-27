"""
Content-Based Workout Recommender
=================================

Pipeline (matches the project PPT):

    User Input -> User Profile -> Feature Extraction -> Content-Based Filtering
    -> Feature Vectors -> Cosine Similarity -> Similarity Score -> Ranking
    -> Top-N Recommendations -> Personalized Workout

1. FEATURE EXTRACTION (exercise side, done once in `FeatureEncoder.fit`)
   Each exercise becomes a vector made of 8 feature blocks:

   block       encoding                                   weight
   ----------  -----------------------------------------  ------
   goal        multi-hot (primary=1.0, secondary=0.6)       3.0
   muscle      one-hot over 10 muscle groups                3.0
   equipment   8 equipment characteristics (bodyweight,     2.5
               free-weight, machine/guided, elastic, ...)
   level       one-hot over difficulty                      2.5
   type        one-hot over 5 exercise types                1.5
   duration    soft (Gaussian) bins over exercise minutes   1.0
   difficulty  one-hot over difficulty                      1.0
   text        TF-IDF of the exercise keyword profile       1.0

2. USER PROFILE REPRESENTATION (`FeatureEncoder.encode_user`)
   The user's form answers are mapped into the SAME feature space, e.g.
   target "Chest" -> muscle block with Chest=1.0, Triceps=0.35, ...;
   "Full Gym" -> equipment preference vector; goal + type -> TF-IDF query.

3. Each block is L2-normalised and scaled by sqrt(weight), then all blocks
   are concatenated. Cosine similarity of the concatenated vectors is

        cos(U, E) = (U . E) / (||U|| x ||E||)
                  = sum_b w_b cos(U_b, E_b) / (||U|| ||E||)

   i.e. a weighted combination of per-block similarities. The per-block
   terms are kept so the app can explain every score.

4. HARD ELIGIBILITY FILTER with a relaxation cascade runs BEFORE ranking.
5. Eligible exercises are RANKED by cosine similarity -> Top-N.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from model.config import (BLOCK_LABELS, CONDITIONING_GOALS, COOLDOWN_MINUTES,
                          EQUIPMENT_ACCESS, EQUIPMENT_ATTRIBUTES,
                          EXERCISE_COUNT_RANGE, EXERCISE_EQUIPMENT,
                          EXERCISE_EQUIPMENT_PROFILE,
                          EXERCISE_TYPES, FEATURE_WEIGHTS, FITNESS_LEVELS,
                          GOAL_COMPATIBLE_TYPES, GOAL_KEYWORDS,
                          GOAL_TYPE_AFFINITY, GOAL_TYPE_BLEND, GOALS,
                          LEVEL_AFFINITY, LEVEL_ALLOWED_RELAXED,
                          LEVEL_ALLOWED_STRICT, MUSCLE_AFFINITY, MUSCLE_GROUPS,
                          MUSCLE_KEYWORDS, SECONDARY_GOAL_WEIGHT,
                          SYNERGIST_THRESHOLD, TYPE_AFFINITY, TYPE_KEYWORDS,
                          USER_EQUIPMENT_PROFILE, USER_GOAL_AFFINITY,
                          WARMUP_MINUTES)
from model.user_profile import UserProfile
from utils.preprocessing import DEFAULT_DATASET_PATH, load_exercise_data

# Duration encoding: Gaussian kernels on log(minutes) centred at these values
DURATION_CENTRES = np.array([2.5, 4.0, 6.0, 9.0, 13.0, 18.0])
DURATION_SIGMA = 0.35

BLOCK_ORDER = ["goal", "muscle", "equipment", "level", "type", "duration",
               "difficulty", "text"]

FILTER_STAGES = [
    "Strict match",
    "Relaxed difficulty",
    "Relaxed secondary constraints",
    "Broad (equipment only)",
]


# ----------------------------------------------------------------------------
# Small vector helpers
# ----------------------------------------------------------------------------
def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation; all-zero rows stay zero."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return np.divide(mat, norms, out=np.zeros_like(mat), where=norms > 0)


def _vector_from_mapping(mapping: dict[str, float], vocab: list[str]) -> np.ndarray:
    return np.array([[mapping.get(v, 0.0) for v in vocab]], dtype=float)


def _duration_kernel(minutes: np.ndarray) -> np.ndarray:
    """Soft binning of minutes -> (n, len(DURATION_CENTRES))."""
    m = np.log(np.clip(np.asarray(minutes, dtype=float), 0.5, None))[:, None]
    c = np.log(DURATION_CENTRES)[None, :]
    return np.exp(-((m - c) ** 2) / (2 * DURATION_SIGMA ** 2))


def per_exercise_time_budget(duration: int) -> float:
    """Minutes available per exercise for a session length (used in profile)."""
    lo, hi = EXERCISE_COUNT_RANGE.get(duration, (4, 5))
    usable = duration - WARMUP_MINUTES.get(duration, 5) - COOLDOWN_MINUTES.get(duration, 3)
    return max(usable / ((lo + hi) / 2.0), 1.0)


def build_user_query(profile: UserProfile) -> str:
    """Keyword 'document' that describes the user, for the TF-IDF block."""
    parts = [GOAL_KEYWORDS[profile.goal], MUSCLE_KEYWORDS[profile.target_muscle],
             TYPE_KEYWORDS[profile.exercise_type]]
    if profile.goal in CONDITIONING_GOALS:
        parts.append(MUSCLE_KEYWORDS["Cardio"])
    if profile.equipment == "No Equipment":
        parts.append("bodyweight")
    return " ".join(parts)


# ----------------------------------------------------------------------------
# Feature encoder
# ----------------------------------------------------------------------------
class FeatureEncoder:
    """Turns exercises and user profiles into vectors in the same space."""

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = dict(FEATURE_WEIGHTS if weights is None else weights)
        self.vocab: dict[str, list[str]] = {
            "goal": list(GOALS), "muscle": list(MUSCLE_GROUPS),
            "equipment": list(EQUIPMENT_ATTRIBUTES), "level": list(FITNESS_LEVELS),
            "type": list(EXERCISE_TYPES),
            "duration": [f"~{c:g} min" for c in DURATION_CENTRES],
            "difficulty": list(FITNESS_LEVELS),
        }
        self.tfidf = TfidfVectorizer(token_pattern=r"[a-z][a-z\-]+", sublinear_tf=True,
                                     min_df=1)
        self.blocks: dict[str, np.ndarray] = {}
        self.matrix: np.ndarray | None = None
        self._fitted = False

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        n = len(df)
        raw: dict[str, np.ndarray] = {}

        # goal: multi-hot, primary goal weighted higher
        goal_idx = {g: i for i, g in enumerate(self.vocab["goal"])}
        g = np.zeros((n, len(goal_idx)))
        for row, goals in enumerate(df["goals_list"]):
            for k, goal in enumerate(goals):
                g[row, goal_idx[goal]] = 1.0 if k == 0 else SECONDARY_GOAL_WEIGHT
        raw["goal"] = g

        # one-hot categorical blocks (vectorised with pandas Categorical codes)
        # equipment: described by characteristics (see config.EQUIPMENT_ATTRIBUTES)
        equip_lookup = {e: _vector_from_mapping(EXERCISE_EQUIPMENT_PROFILE[e], self.vocab["equipment"])[0]
                        for e in EXERCISE_EQUIPMENT}
        raw["equipment"] = np.vstack([equip_lookup[e] for e in df["equipment"]])

        for block, col in (("muscle", "muscle_group"),
                           ("level", "difficulty"), ("type", "exercise_type"),
                           ("difficulty", "difficulty")):
            codes = pd.Categorical(df[col], categories=self.vocab[block]).codes
            onehot = np.zeros((n, len(self.vocab[block])))
            valid = codes >= 0
            onehot[np.flatnonzero(valid), codes[valid]] = 1.0
            raw[block] = onehot

        raw["duration"] = _duration_kernel(df["duration_minutes"].to_numpy())
        raw["text"] = self.tfidf.fit_transform(df["tags"]).toarray()
        self.vocab["text"] = list(self.tfidf.get_feature_names_out())

        self.blocks = {b: _l2_normalise(raw[b]) for b in BLOCK_ORDER}
        self.matrix = self._assemble(self.blocks)
        self._fitted = True
        return self

    def _assemble(self, blocks: dict[str, np.ndarray]) -> np.ndarray:
        return np.hstack([np.sqrt(self.weights[b]) * blocks[b] for b in BLOCK_ORDER])

    # ----------------------------------------------------------- user side
    def encode_user_blocks(self, profile: UserProfile) -> dict[str, np.ndarray]:
        """User profile -> normalised feature blocks (1 x d_b each)."""
        if not self._fitted:
            raise RuntimeError("FeatureEncoder must be fitted before encoding users.")
        raw: dict[str, np.ndarray] = {}
        raw["goal"] = _vector_from_mapping(USER_GOAL_AFFINITY[profile.goal], self.vocab["goal"])
        raw["muscle"] = _vector_from_mapping(self._muscle_preferences(profile), self.vocab["muscle"])
        raw["equipment"] = _vector_from_mapping(USER_EQUIPMENT_PROFILE[profile.equipment],
                                                self.vocab["equipment"])
        raw["level"] = _vector_from_mapping({profile.fitness_level: 1.0}, self.vocab["level"])
        raw["type"] = _vector_from_mapping(self._type_preferences(profile), self.vocab["type"])
        raw["duration"] = _duration_kernel(np.array([per_exercise_time_budget(profile.duration)]))
        raw["difficulty"] = _vector_from_mapping(LEVEL_AFFINITY[profile.fitness_level],
                                                 self.vocab["difficulty"])
        raw["text"] = self.tfidf.transform([build_user_query(profile)]).toarray()
        return {b: _l2_normalise(raw[b]) for b in BLOCK_ORDER}

    def encode_user(self, profile: UserProfile) -> np.ndarray:
        return self._assemble(self.encode_user_blocks(profile))

    @staticmethod
    def _muscle_preferences(profile: UserProfile) -> dict[str, float]:
        prefs = dict(MUSCLE_AFFINITY[profile.target_muscle])
        if profile.goal in CONDITIONING_GOALS:
            # Conditioning goals value full-body and cardio work for any target
            prefs["Full Body"] = max(prefs.get("Full Body", 0.0), 0.5)
            prefs["Cardio"] = max(prefs.get("Cardio", 0.0), 0.45)
        return prefs

    @staticmethod
    def _type_preferences(profile: UserProfile) -> dict[str, float]:
        pref = TYPE_AFFINITY[profile.exercise_type]
        goal = GOAL_TYPE_AFFINITY[profile.goal]
        return {t: (1 - GOAL_TYPE_BLEND) * pref.get(t, 0.0) + GOAL_TYPE_BLEND * goal.get(t, 0.0)
                for t in EXERCISE_TYPES}

    def feature_names(self) -> list[str]:
        return [f"{b}:{v}" for b in BLOCK_ORDER for v in self.vocab[b]]


# ----------------------------------------------------------------------------
# Results container
# ----------------------------------------------------------------------------
@dataclass
class RecommendationResult:
    profile: UserProfile
    ranked: pd.DataFrame            # all eligible exercises, best first
    top_n: pd.DataFrame             # first N rows of `ranked`
    filter_stage: str
    filter_stage_index: int
    n_eligible: int
    user_vector: np.ndarray
    warnings: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Recommender
# ----------------------------------------------------------------------------
class WorkoutRecommender:
    """Content-based recommender: hard filter -> cosine similarity -> Top-N."""

    def __init__(self, exercises: pd.DataFrame, weights: dict[str, float] | None = None):
        self.df = exercises.reset_index(drop=True)
        self.encoder = FeatureEncoder(weights).fit(self.df)
        self.X = self.encoder.matrix
        self._X_norm = np.linalg.norm(self.X, axis=1)
        # Cached numpy arrays for fast boolean filtering
        self._equip = self.df["equipment"].to_numpy()
        self._diff = self.df["difficulty"].to_numpy()
        self._muscle = self.df["muscle_group"].to_numpy()
        self._type = self.df["exercise_type"].to_numpy()

    @classmethod
    def from_csv(cls, path: str | Path = DEFAULT_DATASET_PATH) -> "WorkoutRecommender":
        df, _ = load_exercise_data(path)
        return cls(df)

    # -------------------------------------------------- hard eligibility filter
    def eligibility_mask(self, profile: UserProfile, stage: int) -> np.ndarray:
        """
        Boolean mask of exercises allowed at a given relaxation stage.

        Equipment is a physical constraint and is NEVER relaxed.

        stage 0  strict:   equipment + level(strict) + muscle(target) + goal-compatible type
        stage 1  relaxed difficulty: level(relaxed)
        stage 2  relaxed secondary:  + synergist muscles (e.g. triceps for chest)
        stage 3  broad:    equipment only (ranking still prefers the best fits)
        """
        mask = np.isin(self._equip, list(EQUIPMENT_ACCESS[profile.equipment]))
        if stage >= 3:
            return mask

        levels = (LEVEL_ALLOWED_STRICT if stage == 0 else LEVEL_ALLOWED_RELAXED)[profile.fitness_level]
        mask &= np.isin(self._diff, list(levels))

        muscles = {profile.target_muscle}
        if profile.target_muscle == "Full Body" or profile.goal in CONDITIONING_GOALS:
            muscles |= {"Full Body", "Cardio"}
        if stage >= 2:
            muscles |= {m for m, a in MUSCLE_AFFINITY[profile.target_muscle].items()
                        if a >= SYNERGIST_THRESHOLD}
        mask &= np.isin(self._muscle, list(muscles))

        if stage <= 2:
            types = set(GOAL_COMPATIBLE_TYPES[profile.goal])
            if profile.exercise_type != "Mixed":
                types.add(profile.exercise_type)
            mask &= np.isin(self._type, list(types))
        return mask

    def filter_candidates(self, profile: UserProfile, required: int) -> tuple[np.ndarray, int]:
        """Apply the cascade until at least `required` exercises are eligible."""
        mask = np.zeros(len(self.df), dtype=bool)
        for stage in range(len(FILTER_STAGES)):
            mask = self.eligibility_mask(profile, stage)
            if mask.sum() >= required:
                return mask, stage
        return mask, len(FILTER_STAGES) - 1

    # ------------------------------------------------------------- similarity
    def similarity_scores(self, user_vec: np.ndarray) -> np.ndarray:
        """Cosine similarity between the user vector and EVERY exercise."""
        return cosine_similarity(user_vec, self.X).ravel()

    def block_similarities(self, profile: UserProfile) -> dict[str, np.ndarray]:
        """Per-block cosine similarity cos(U_b, E_b) for every exercise."""
        return self._score(profile)[2]

    def score_contributions(self, profile: UserProfile, user_vec: np.ndarray) -> dict[str, np.ndarray]:
        """
        Exact decomposition of the cosine score into block contributions:
            contribution_b = w_b * cos_b / (||U|| * ||E||)     (sums to the score)
        """
        block_cos = self.block_similarities(profile)
        denom = np.linalg.norm(user_vec) * self._X_norm
        denom = np.where(denom > 0, denom, 1.0)
        return {b: self.encoder.weights[b] * block_cos[b] / denom for b in BLOCK_ORDER}

    # -------------------------------------------------------------- scoring
    def _score(self, profile: UserProfile):
        """Encode the user once and compute cosine + per-block similarities."""
        user_blocks = self.encoder.encode_user_blocks(profile)
        user_vec = self.encoder._assemble(user_blocks)
        scores = self.similarity_scores(user_vec)          # cosine similarity
        block_cos = {b: self.encoder.blocks[b] @ user_blocks[b].ravel() for b in BLOCK_ORDER}
        return user_vec, scores, block_cos

    def _rank(self, mask: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Row indices of eligible exercises, best score first (ties -> exercise_id)."""
        idx = np.flatnonzero(mask)
        return idx[np.lexsort((self.df["exercise_id"].to_numpy()[idx], -scores[idx]))]

    def rank_indices(self, profile: UserProfile, top_n: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Fast path (no DataFrame building) used by the evaluation module:
        returns (row indices of the Top-N, their cosine scores).
        """
        mask, _ = self.filter_candidates(profile, top_n)
        _, scores, _ = self._score(profile)
        order = self._rank(mask, scores)[:top_n]
        return order, scores[order]

    # -------------------------------------------------------------- recommend
    def recommend(self, profile: UserProfile, top_n: int = 5,
                  min_candidates: int | None = None) -> RecommendationResult:
        """
        Full pipeline for one user: filter -> cosine similarity -> ranking -> Top-N.

        `min_candidates` lets the workout generator ask for a larger eligible
        pool (it needs extra options for diversity / regeneration).
        """
        top_n = int(max(1, top_n))
        required = max(top_n, min_candidates or 0)
        mask, stage = self.filter_candidates(profile, required)

        user_vec, scores, block_cos = self._score(profile)
        denom = np.linalg.norm(user_vec) * self._X_norm
        denom = np.where(denom > 0, denom, 1.0)
        order = self._rank(mask, scores)

        # Build all score columns at once (much faster than inserting one by one)
        extra = {"rank": np.arange(1, len(order) + 1),
                 "similarity": scores[order],
                 "match_percent": np.round(scores[order] * 100, 1)}
        for b in BLOCK_ORDER:
            extra[f"sim_{b}"] = block_cos[b][order]
            # exact decomposition: contribution_b = w_b * cos_b / (||U|| ||E||)
            extra[f"contrib_{b}"] = self.encoder.weights[b] * block_cos[b][order] / denom[order]
        base = self.df.iloc[order].reset_index().rename(columns={"index": "row_index"})
        ranked = pd.concat([pd.DataFrame(extra), base], axis=1)

        warnings: list[str] = []
        if stage > 0:
            warnings.append(f"Only a few exercises matched every preference, so the filter was "
                            f"relaxed to '{FILTER_STAGES[stage]}'.")
        if len(ranked) < top_n:
            warnings.append(f"Only {len(ranked)} exercise(s) are possible with your equipment; "
                            f"showing all of them.")

        return RecommendationResult(
            profile=profile, ranked=ranked, top_n=ranked.head(top_n).copy(),
            filter_stage=FILTER_STAGES[stage], filter_stage_index=stage,
            n_eligible=int(mask.sum()), user_vector=user_vec, warnings=warnings,
        )

    # ------------------------------------------------------- utilities
    def exercise_similarity(self, rows_a: np.ndarray, rows_b: np.ndarray) -> np.ndarray:
        """Cosine similarity between exercises (used for workout diversity)."""
        return cosine_similarity(self.X[rows_a], self.X[rows_b])

    @staticmethod
    def block_labels() -> dict[str, str]:
        return dict(BLOCK_LABELS)
