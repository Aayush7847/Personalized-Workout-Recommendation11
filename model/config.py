"""
Central configuration for the recommendation engine.

Every vocabulary, weight and rule table used by the recommender, the
workout generator, the Streamlit app, the notebook and the evaluation
module lives here, so all of them stay consistent (single source of truth).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Vocabularies (user-facing options + dataset categories)
# ---------------------------------------------------------------------------
FITNESS_LEVELS = ["Beginner", "Intermediate", "Advanced"]

GOALS = ["Weight Loss", "Muscle Gain", "Strength", "General Fitness", "Endurance"]

TARGET_MUSCLES = ["Chest", "Back", "Shoulders", "Biceps", "Triceps",
                  "Legs", "Glutes", "Core", "Full Body"]

# Dataset muscle groups = user targets + a dedicated "Cardio" group
MUSCLE_GROUPS = TARGET_MUSCLES + ["Cardio"]

USER_EQUIPMENT = ["No Equipment", "Dumbbells", "Barbell", "Resistance Bands",
                  "Home Gym", "Full Gym"]

EXERCISE_EQUIPMENT = ["No Equipment", "Dumbbells", "Barbell", "Resistance Band",
                      "Cable", "Machine", "Bench", "Kettlebell", "Treadmill",
                      "Pull-up Bar", "Full Gym"]

DURATIONS = [20, 30, 45, 60, 90]
WORKOUT_DAYS = [2, 3, 4, 5, 6]

EXERCISE_TYPES = ["Strength", "Cardio", "HIIT", "Mobility", "Core"]
USER_EXERCISE_TYPES = EXERCISE_TYPES + ["Mixed"]

TOP_N_OPTIONS = [5, 8, 10]

# ---------------------------------------------------------------------------
# Feature weights (conceptual importance from the project brief)
#
# The final vector is a concatenation of feature *blocks*. Each block is
# L2-normalised and then multiplied by sqrt(weight). Because of this, the
# cosine similarity of the full vectors equals a weighted average of the
# per-block cosine similarities:
#
#     cos(U, E) = sum_b  w_b * cos(U_b, E_b)  /  sum_b w_b
#
# (exactly, whenever every block is non-zero). So a weight of 3.0 means that
# block counts three times as much as a block with weight 1.0.
# ---------------------------------------------------------------------------
FEATURE_WEIGHTS: dict[str, float] = {
    "goal": 3.0,         # high   - what the user is training for
    "muscle": 3.0,       # high   - target muscle group
    "equipment": 2.5,    # high   - equipment preference / setup
    "level": 2.5,        # high   - exact fitness-level match
    "type": 1.5,         # medium - preferred exercise type
    "duration": 1.0,     # medium - time per exercise vs session budget
    "difficulty": 1.0,   # medium - tolerance for adjacent difficulty levels
    "text": 1.0,         # medium - TF-IDF keyword profile (movement description)
}

BLOCK_LABELS = {
    "goal": "Goal", "muscle": "Target muscle", "equipment": "Equipment",
    "level": "Fitness level", "type": "Exercise type", "duration": "Duration",
    "difficulty": "Difficulty proximity", "text": "Keyword profile (TF-IDF)",
}

# Weight of an exercise's secondary goals relative to its primary goal
SECONDARY_GOAL_WEIGHT = 0.6

# ---------------------------------------------------------------------------
# HARD ELIGIBILITY RULES (applied before similarity)
# ---------------------------------------------------------------------------
# Which exercise equipment each user setup can physically perform.
EQUIPMENT_ACCESS: dict[str, set[str]] = {
    "No Equipment": {"No Equipment"},
    "Dumbbells": {"No Equipment", "Dumbbells"},
    "Barbell": {"No Equipment", "Barbell", "Bench"},
    "Resistance Bands": {"No Equipment", "Resistance Band"},
    "Home Gym": {"No Equipment", "Dumbbells", "Barbell", "Resistance Band",
                 "Bench", "Kettlebell", "Pull-up Bar"},
    "Full Gym": set(EXERCISE_EQUIPMENT),
}

# Equipment FEATURE REPRESENTATION.
# Instead of a plain one-hot (which makes "Full Gym" users equally far from
# every individual machine), each piece of equipment is described by a few
# characteristics. Users and exercises are compared on these characteristics.
EQUIPMENT_ATTRIBUTES = ["bodyweight", "free_weight", "external_load", "guided",
                        "elastic", "cardio_machine", "portable", "gym_station"]

EXERCISE_EQUIPMENT_PROFILE: dict[str, dict[str, float]] = {
    "No Equipment": {"bodyweight": 1.0, "portable": 1.0},
    "Dumbbells": {"free_weight": 1.0, "external_load": 1.0, "portable": 0.5},
    "Barbell": {"free_weight": 1.0, "external_load": 1.0},
    "Resistance Band": {"elastic": 1.0, "portable": 1.0, "external_load": 0.4},
    "Cable": {"external_load": 1.0, "guided": 0.6, "gym_station": 1.0},
    "Machine": {"external_load": 1.0, "guided": 1.0, "gym_station": 1.0},
    "Bench": {"bodyweight": 1.0, "portable": 0.3},
    "Kettlebell": {"free_weight": 1.0, "external_load": 1.0, "portable": 0.6},
    "Treadmill": {"cardio_machine": 1.0, "gym_station": 0.6},
    "Pull-up Bar": {"bodyweight": 1.0, "external_load": 0.2},
    "Full Gym": {"gym_station": 1.0, "external_load": 0.7, "cardio_machine": 0.3},
}

# What each user setup "looks like" in the same characteristic space.
USER_EQUIPMENT_PROFILE: dict[str, dict[str, float]] = {
    "No Equipment": {"bodyweight": 1.0, "portable": 1.0},
    "Dumbbells": {"free_weight": 1.0, "external_load": 1.0, "portable": 0.5, "bodyweight": 0.3},
    "Barbell": {"free_weight": 1.0, "external_load": 1.0, "bodyweight": 0.3},
    "Resistance Bands": {"elastic": 1.0, "portable": 1.0, "external_load": 0.4, "bodyweight": 0.3},
    "Home Gym": {"free_weight": 1.0, "external_load": 1.0, "portable": 0.4,
                 "bodyweight": 0.4, "elastic": 0.2},
    "Full Gym": {"external_load": 1.0, "free_weight": 0.8, "guided": 0.8,
                 "gym_station": 0.8, "cardio_machine": 0.3, "bodyweight": 0.2},
}

# Goal FEATURE REPRESENTATION for the user: related goals get partial credit,
# so exercises that serve several compatible goals (e.g. Bench Press:
# Muscle Gain + Strength) are not penalised for being versatile.
USER_GOAL_AFFINITY: dict[str, dict[str, float]] = {
    "Weight Loss": {"Weight Loss": 1.0, "Endurance": 0.5, "General Fitness": 0.3},
    "Muscle Gain": {"Muscle Gain": 1.0, "Strength": 0.5},
    "Strength": {"Strength": 1.0, "Muscle Gain": 0.45},
    "General Fitness": {"General Fitness": 1.0, "Weight Loss": 0.3, "Endurance": 0.3,
                        "Muscle Gain": 0.25, "Strength": 0.25},
    "Endurance": {"Endurance": 1.0, "Weight Loss": 0.5, "General Fitness": 0.3},
}

# Difficulty levels allowed at each relaxation stage of the filter cascade.
LEVEL_ALLOWED_STRICT = {
    "Beginner": {"Beginner"},
    "Intermediate": {"Beginner", "Intermediate"},
    "Advanced": {"Intermediate", "Advanced"},
}
LEVEL_ALLOWED_RELAXED = {
    "Beginner": {"Beginner", "Intermediate"},
    "Intermediate": {"Beginner", "Intermediate", "Advanced"},
    "Advanced": {"Beginner", "Intermediate", "Advanced"},
}

# Soft affinity of a user level for each exercise difficulty
# (used for the "difficulty proximity" feature block).
LEVEL_AFFINITY = {
    "Beginner": {"Beginner": 1.0, "Intermediate": 0.35, "Advanced": 0.0},
    "Intermediate": {"Beginner": 0.45, "Intermediate": 1.0, "Advanced": 0.4},
    "Advanced": {"Beginner": 0.15, "Intermediate": 0.6, "Advanced": 1.0},
}

# Exercise types that are appropriate for each goal (strict stage only).
GOAL_COMPATIBLE_TYPES: dict[str, set[str]] = {
    "Weight Loss": {"Cardio", "HIIT", "Strength", "Core"},
    "Muscle Gain": {"Strength", "Core"},
    "Strength": {"Strength", "Core"},
    "General Fitness": set(EXERCISE_TYPES),
    "Endurance": {"Cardio", "HIIT", "Strength", "Core"},
}

# Goals for which full-body and cardio work is always eligible / prioritised.
CONDITIONING_GOALS = {"Weight Loss", "Endurance"}

# Muscle affinity: how much training muscle group X helps target Y.
# Used for (a) the muscle feature block and (b) relaxed filtering.
MUSCLE_AFFINITY: dict[str, dict[str, float]] = {
    "Chest": {"Chest": 1.0, "Triceps": 0.35, "Shoulders": 0.3, "Full Body": 0.2},
    "Back": {"Back": 1.0, "Biceps": 0.35, "Shoulders": 0.2, "Full Body": 0.2},
    "Shoulders": {"Shoulders": 1.0, "Chest": 0.25, "Triceps": 0.25, "Back": 0.2,
                  "Full Body": 0.2},
    "Biceps": {"Biceps": 1.0, "Back": 0.4, "Full Body": 0.1},
    "Triceps": {"Triceps": 1.0, "Chest": 0.4, "Shoulders": 0.3},
    "Legs": {"Legs": 1.0, "Glutes": 0.5, "Full Body": 0.3, "Cardio": 0.1},
    "Glutes": {"Glutes": 1.0, "Legs": 0.5, "Full Body": 0.2},
    "Core": {"Core": 1.0, "Full Body": 0.3, "Back": 0.1},
    "Full Body": {"Full Body": 1.0, "Cardio": 0.4, "Legs": 0.25, "Core": 0.2,
                  "Back": 0.15, "Chest": 0.15, "Glutes": 0.15, "Shoulders": 0.15},
}
SYNERGIST_THRESHOLD = 0.3  # affinity needed to be eligible in relaxed stage

# Preferred exercise type -> affinity over dataset exercise types.
TYPE_AFFINITY: dict[str, dict[str, float]] = {
    "Strength": {"Strength": 1.0, "Core": 0.25},
    "Cardio": {"Cardio": 1.0, "HIIT": 0.5},
    "HIIT": {"HIIT": 1.0, "Cardio": 0.45, "Strength": 0.1},
    "Mobility": {"Mobility": 1.0, "Core": 0.3},
    "Core": {"Core": 1.0, "Mobility": 0.2},
    "Mixed": {"Strength": 0.8, "Cardio": 0.6, "HIIT": 0.6, "Core": 0.6,
              "Mobility": 0.35},
}

# Goals also shape which types are favoured (blended into the type block).
GOAL_TYPE_AFFINITY: dict[str, dict[str, float]] = {
    "Weight Loss": {"HIIT": 1.0, "Cardio": 0.9, "Strength": 0.45, "Core": 0.35},
    "Muscle Gain": {"Strength": 1.0, "Core": 0.25},
    "Strength": {"Strength": 1.0, "Core": 0.3},
    "General Fitness": {"Strength": 0.7, "Cardio": 0.6, "Core": 0.6,
                        "HIIT": 0.5, "Mobility": 0.5},
    "Endurance": {"Cardio": 1.0, "HIIT": 0.8, "Strength": 0.4, "Core": 0.4},
}
GOAL_TYPE_BLEND = 0.5  # 50% preferred type, 50% goal-driven type affinity

# Goal -> descriptive keywords used to build the TF-IDF "query" of a user.
GOAL_KEYWORDS = {
    "Weight Loss": "metabolic conditioning explosive full-body circuit calorie-burn",
    "Muscle Gain": "hypertrophy compound isolation controlled tension",
    "Strength": "heavy compound maximal-force power barbell",
    "General Fitness": "functional balanced stability mobility bodyweight",
    "Endurance": "aerobic stamina high-rep conditioning sustained",
}
MUSCLE_KEYWORDS = {
    "Chest": "chest pectorals push press",
    "Back": "back lats rhomboids pull row",
    "Shoulders": "shoulders deltoids overhead raise",
    "Biceps": "biceps arms curl elbow-flexion",
    "Triceps": "triceps arms extension elbow-extension",
    "Legs": "legs quadriceps hamstrings calves squat",
    "Glutes": "glutes hips hip-extension bridge",
    "Core": "core abdominals obliques stability",
    "Full Body": "full-body total-body compound explosive",
    "Cardio": "cardio aerobic heart-rate",
}
TYPE_KEYWORDS = {
    "Strength": "resistance strength",
    "Cardio": "aerobic cardio steady",
    "HIIT": "interval explosive metabolic",
    "Mobility": "mobility stretch flexibility",
    "Core": "core stability abdominals",
    "Mixed": "functional",
}

# ---------------------------------------------------------------------------
# Workout-generation settings
# ---------------------------------------------------------------------------
# Number-of-exercises range for each session length (minutes)
EXERCISE_COUNT_RANGE = {20: (3, 4), 30: (3, 4), 45: (4, 5), 60: (5, 6), 90: (6, 8)}
WARMUP_MINUTES = {20: 3, 30: 4, 45: 5, 60: 6, 90: 10}
COOLDOWN_MINUTES = {20: 2, 30: 3, 45: 4, 60: 4, 90: 6}

SECONDS_PER_REP = 4          # ~controlled tempo
TRANSITION_SECONDS = 60      # moving to / setting up the next exercise
MAX_SETS = 5
MIN_SETS = 2

# Goal-driven rep prescriptions for repetition-based (non-timed) exercises.
#   (sets, reps text, rest seconds) for compound and isolation movements
GOAL_PRESCRIPTION = {
    "Strength":        {"compound": (4, "4-6", 150), "isolation": (3, "6-8", 90)},
    "Muscle Gain":     {"compound": (4, "8-10", 90), "isolation": (3, "10-12", 60)},
    "General Fitness": {"compound": (3, "10-12", 60), "isolation": (3, "12", 45)},
    "Endurance":       {"compound": (3, "15-20", 40), "isolation": (3, "15-20", 30)},
    "Weight Loss":     {"compound": (3, "12-15", 40), "isolation": (3, "15", 30)},
}

# Calories per minute in the dataset are for a 70 kg reference person; the
# estimate is scaled linearly by body weight. Values are approximate.
REFERENCE_BODY_WEIGHT_KG = 70.0
WARMUP_KCAL_PER_MIN = 4.0

DISCLAIMER = (
    "This project is developed for educational purposes. Workout recommendations "
    "are general informational suggestions and are not medical advice or a "
    "substitute for a qualified fitness professional. Users with injuries, medical "
    "conditions, or other health concerns should consult an appropriate healthcare "
    "professional before exercising."
)

REQUIRED_COLUMNS = [
    "exercise_id", "exercise_name", "muscle_group", "equipment", "difficulty",
    "goal", "exercise_type", "duration_minutes", "calories_burned", "sets",
    "reps", "rest_seconds", "instructions", "safety_notes",
]
# Extra (optional) columns produced by the dataset generator
OPTIONAL_COLUMNS = ["mechanics", "movement_pattern", "tags"]
