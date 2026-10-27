# Personalized Workout Recommendation System Using Machine Learning & Recommendation Systems

**PERSONALIZED WORKOUT — AI-Assisted Recommendation System**

Mini-project · Subject: Recommendation Systems · Electronics & Computer Science, 2026–27
Author: Aayush Deshmukh

> ⚠️ This project is developed for educational purposes. Workout recommendations are general informational suggestions and are not medical advice or a substitute for a qualified fitness professional. Users with injuries, medical conditions, or other health concerns should consult an appropriate healthcare professional before exercising.

---

## Table of Contents
1. [Abstract](#abstract)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Features](#features)
5. [Tech Stack](#tech-stack)
6. [System Architecture](#system-architecture)
7. [Dataset](#dataset)
8. [Algorithm](#algorithm)
9. [Content-Based Filtering](#content-based-filtering)
10. [Cosine Similarity](#cosine-similarity)
11. [Feature Engineering](#feature-engineering)
12. [Workflow](#workflow)
13. [Installation](#installation)
14. [How to Run](#how-to-run)
15. [Launch Website](#launch-website)
16. [How to Use](#how-to-use)
17. [Example User](#example-user)
18. [Example Recommendation](#example-recommendation)
19. [Evaluation](#evaluation)
20. [Project Structure](#project-structure)
21. [Limitations](#limitations)
22. [Future Scope](#future-scope)
23. [Conclusion](#conclusion)
24. [Disclaimer](#disclaimer)

---

## Abstract
Most workout plans are generic: they ignore a person's goal, fitness level, available equipment
and time. This project builds a **content-based recommendation system** that represents both
the user and every exercise as **weighted feature vectors** in the same space and ranks exercises
by **cosine similarity**. A hard eligibility filter keeps results safe and practical (e.g. no
barbell exercises for a user without equipment), and a workout generator turns the ranked list
into a complete, time-fitted session with sets, reps and rest. Every score is explainable: it
splits exactly into contributions from goal, muscle, equipment, level and other features. The
system runs locally as a Streamlit web app on a curated dataset of 298 exercises.

## Problem Statement
People who want to train often do not know which exercises suit their **goal** (weight loss,
muscle gain, strength, general fitness, endurance), **target muscle group**, **fitness level**,
**available equipment**, **time** and **preferred training type**. Generic plans lead to poor
results, boredom or unsafe choices. The problem is to automatically recommend relevant exercises
and assemble a personalised workout from a user's inputs — like Netflix recommends films or
Spotify recommends songs, but for exercise.

## Objectives
1. Collect user details and fitness preferences through a simple web interface.
2. Build a user profile and represent it as a feature vector.
3. Represent each exercise as a feature vector (categorical + numeric + TF-IDF text features).
4. Use **content-based filtering** and **cosine similarity** to score exercises.
5. Rank exercises and return **Top-N recommendations** with match percentages.
6. Generate a complete personalised workout (exercises, sets, reps, rest, time).
7. Explain every recommendation ("Why this exercise?") and evaluate the system with Precision@K / Recall@K.

## Features
- 🧍 **User profile input** — age, height, weight, fitness level, goal, target muscle, equipment, duration (20/30/45/60/90 min), days per week (2–6), preferred type (Strength / Cardio / HIIT / Mobility / Core / Mixed).
- 📏 **BMI** with WHO category — informational only, never used to choose exercises.
- 🧠 **Content-based recommender** — 8 weighted feature blocks, TF-IDF keyword profile, cosine similarity.
- 🚧 **Hard eligibility filter with a 4-stage fallback cascade** — results are never empty; equipment is never relaxed.
- ⭐ **Top-N recommendations** (5 / 8 / 10) with rank, match %, muscle, equipment, difficulty and type.
- 💡 **"Why this exercise?"** — dynamic ✓ / ◐ / ⚠ reasons plus an exact per-feature score breakdown and radar chart.
- 🗓️ **Workout generator** — duration-based exercise count, goal-based sets/reps/rest, time estimate, diversity (MMR), warm-up and cool-down.
- 🔄 **Generate Another Workout** — same constraints, different high-ranked exercises.
- 📈 **Workout statistics** — dashboard cards, time donut, per-exercise minutes and calories, weekly projection.
- 🕘 **History** saved to `data/workout_history.json` (view, download, clear).
- 💬 **Feedback** (1–5 rating, 👍/👎, comment) saved to `data/feedback.json`.
- 📊 **Dataset Explorer** — totals, filters, search, download, 6 analytics charts.
- 🧪 **Evaluation tab** — Precision@K, Recall@K, nDCG@K and a cosine check (clearly marked as demonstration).
- ℹ️ **How It Works tab** — pipeline, formula, weights, filter cascade, and your own profile vector.
- 📓 **Jupyter notebook** with the full analysis, built on the same modules as the app.
- ✅ **16 automated tests** (`python tests/test_project.py`).
- 🛡️ Friendly error handling for invalid inputs, a missing dataset, corrupted JSON files, bad CSV rows and no-match cases.

## Tech Stack
| Layer | Technology |
|---|---|
| Language | Python 3.10+ (tested on 3.12) |
| Web UI | Streamlit |
| Data | Pandas, NumPy, CSV |
| ML / RecSys | scikit-learn (`TfidfVectorizer`, `cosine_similarity`) |
| Charts | Plotly (app), Matplotlib (notebook) |
| Notebook | Jupyter (nbformat / nbconvert) |
| Storage | Local JSON files (history, feedback) |

No external APIs are used, and no internet connection is needed to generate recommendations.

## System Architecture
```
 ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐   ┌───────────────────────┐
 │  User Input  │──▶│ Preprocessing│──▶│   User Profile   │──▶│  Feature Extraction    │
 │ (Streamlit)  │   │ & validation │   │  (UserProfile)   │   │ (FeatureEncoder, TF-IDF)│
 └──────────────┘   └──────────────┘   └──────────────────┘   └──────────┬────────────┘
                                                                          ▼
 ┌──────────────────────┐   ┌─────────┐   ┌─────────────────┐   ┌───────────────────────┐
 │ Personalized Workout │◀──│  Top-N  │◀──│     Ranking     │◀──│ Content-Based Filtering│
 │ (workout_generator)  │   │         │   │ (sort by score) │   │ hard filter + cosine   │
 └──────────────────────┘   └─────────┘   └─────────────────┘   └───────────────────────┘
          │
          ▼
  Explanations · Statistics · History (JSON) · Feedback (JSON)
```

| Module | Responsibility |
|---|---|
| `utils/preprocessing.py` | load CSV, validate columns, clean missing / duplicate / invalid values, derive features |
| `model/user_profile.py` | input validation, `UserProfile`, BMI |
| `model/recommender.py` | `FeatureEncoder` (feature vectors), eligibility filter + cascade, cosine similarity, ranking, Top-N |
| `utils/workout_generator.py` | exercise selection (MMR), sets / reps / rest, time and calorie estimate |
| `utils/explanations.py` | dynamic "Why this exercise?" reasons, score breakdown |
| `utils/storage.py` | safe JSON history / feedback storage with corruption recovery |
| `utils/ui.py`, `utils/charts.py` | styling and Plotly charts |
| `evaluation/evaluate.py` | Precision@K, Recall@K, nDCG@K, cosine verification (demonstration) |

## Dataset
`data/exercises.csv` — **298 unique exercises**, generated by `scripts/generate_dataset.py` from a
hand-curated exercise catalogue (names, muscles, equipment, difficulty, goals, movement patterns).

| Column | Description |
|---|---|
| `exercise_id` | unique id (EX001 …) |
| `exercise_name` | exercise name |
| `muscle_group` | Chest, Back, Shoulders, Biceps, Triceps, Legs, Glutes, Core, Full Body, Cardio |
| `equipment` | No Equipment, Dumbbells, Barbell, Resistance Band, Kettlebell, Machine, Cable, Pull-up Bar, Bench, Treadmill, Full Gym |
| `difficulty` | Beginner (143), Intermediate (105), Advanced (50) |
| `goal` | pipe-separated goals, primary goal first (e.g. `Muscle Gain\|Strength`) |
| `exercise_type` | Strength (188), HIIT (38), Core (31), Mobility (23), Cardio (18) |
| `duration_minutes` | typical time for the prescribed sets |
| `calories_burned` | **approximate, educational** (MET × 3.5 × 70 kg / 200 × minutes) |
| `sets`, `reps`, `rest_seconds` | default prescription |
| `instructions`, `safety_notes` | how to perform the exercise safely |
| `mechanics`, `movement_pattern`, `tags` | extra columns: compound/isolation, movement family, keyword profile (for TF-IDF) |

Muscle-group coverage: Legs 39 · Back 33 · Core 33 · Full Body 33 · Cardio 31 · Chest 30 · Shoulders 28 · Glutes 27 · Biceps 23 · Triceps 21.

User equipment options map to the exercise equipment they unlock: e.g. *Home Gym* = no equipment +
dumbbells + bands + kettlebell + pull-up bar + bench; *Full Gym* = everything.

**Preprocessing** (`clean_exercise_data`): required-column check → category normalisation (case,
spacing, aliases such as "Bodyweight" → "No Equipment") → rows with unknown/missing essential
categories dropped → duplicates removed (by id and name) → invalid numeric values (non-numbers,
negatives, out-of-range) replaced by the median of the same exercise type → missing reps/text
filled → derived features (`goals_list`, `primary_goal`, timed flag, kcal/min, tags).

## Algorithm
```
INPUT: user form answers, exercise dataset
1. Validate the input → UserProfile (BMI computed for information only)
2. Encode every exercise once into a weighted feature vector E (8 blocks)
3. Encode the user into the same feature space → user vector U
4. Hard filter (content-based filtering): keep exercises that are
     • usable with the user's equipment
     • at or below the user's level (strict stage)
     • training the target muscle (or conditioning work for Weight Loss / Endurance)
     • of an exercise type compatible with the goal
   If fewer than needed remain → relax step by step:
     Strict → Relaxed difficulty → Relaxed secondary constraints → Broad (equipment only)
5. score(E) = cos(U, E) = (U·E) / (‖U‖‖E‖) for every eligible exercise
6. Rank by score (ties broken by exercise id) → Top-N recommendations
7. Workout generator: choose a diverse subset with Maximal Marginal Relevance,
   prescribe sets/reps/rest by goal & level, fit the requested duration
OUTPUT: Top-N list with match %, explanations, personalised workout
```

## Content-Based Filtering
Content-based filtering recommends items **similar to what the user wants**, using the **attributes
of the items** (here: exercise goal, muscle, equipment, difficulty, type, duration, keywords) — not
other users' behaviour. This suits a new app with no user history (no cold-start problem for new
users) and makes every recommendation explainable. Collaborative filtering would need many users'
ratings, which this project does not have (see Future Scope).

The system combines two content-based steps:
1. **Rule-based eligibility filtering** — removes exercises that are impossible or unsafe for the user.
2. **Similarity-based ranking** — orders the remaining exercises by cosine similarity.

## Cosine Similarity
$$\cos(A,B)=\frac{A\cdot B}{\lVert A\rVert\times\lVert B\rVert}$$

It measures the angle between two vectors: 1 = same direction (perfect match), 0 = nothing in common.
Because every feature block *b* is L2-normalised and multiplied by √wᵦ before concatenation:

$$\cos(U,E)=\frac{\sum_b w_b\,\cos(U_b,E_b)}{\lVert U\rVert\,\lVert E\rVert}$$

so the final score is a **weighted combination of per-feature similarities**, and the app can show
exactly how many points each feature contributed. The engine uses scikit-learn's
`cosine_similarity`; `evaluation/evaluate.py`, the notebook and the tests verify that it equals the
manual formula.

Match % = cosine × 100 (rounded to 1 decimal). Scores are never random or hard-coded.

## Feature Engineering
| Block | Exercise side | User side | Weight | Priority |
|---|---|---|---|---|
| Goal | multi-hot: primary goal 1.0, secondary goals 0.6 | selected goal 1.0, related goals partial | **3.0** | High |
| Target muscle | one-hot muscle group | target 1.0 + synergists (e.g. Chest → Triceps 0.35) | **3.0** | High |
| Equipment | 8 characteristics (bodyweight, free weight, external load, guided/machine, elastic, cardio machine, portable, gym station) | characteristics of the user's setup | **2.5** | High |
| Fitness level | one-hot difficulty | user level | **2.5** | High |
| Exercise type | one-hot type | preferred type blended 50/50 with the goal's typical types | 1.5 | Medium |
| Duration | Gaussian soft bins on log(minutes) | minutes available per exercise for the chosen session length | 1.0 | Medium |
| Difficulty proximity | one-hot difficulty | graded affinity (own level 1.0, adjacent levels lower) | 1.0 | Medium |
| Keyword profile | **TF-IDF** of exercise tags | TF-IDF of a query built from goal, muscle and type keywords | 1.0 | Medium |

Total vector length: **400 dimensions**. Weights live in `model/config.py → FEATURE_WEIGHTS` and are
explained there and in `model/recommender.py`.

Numeric features (duration) use soft bins instead of raw minutes so that "close" durations are
similar; categorical features use one-hot / multi-hot encoding; text uses TF-IDF.

## Workflow
1. User opens the app and fills in the profile and preferences (or loads a sample profile).
2. Input is validated; errors are shown in plain language.
3. The profile is encoded into a feature vector.
4. The hard filter selects eligible exercises (relaxing if needed, with a notice).
5. Cosine similarity scores → ranking → Top-N table + bar chart.
6. The workout generator builds the session: count by duration (20–30 → 3–4, 45 → 4–5, 60 → 5–6, 90 → 6–8), sets/reps/rest by goal, time estimate = sets × work time + rest between sets + ~1 min set-up per exercise, plus warm-up and cool-down.
7. The user reads the explanations and statistics, can generate an alternative, and can leave feedback.
8. The workout is saved to history automatically.

## Installation
Requirements: **Python 3.10 or newer** and pip.

```bash
# 1. unzip and enter the project folder
cd Personalized_Workout_Recommendation

# 2. (recommended) create a virtual environment
python -m venv .venv
# Windows:     .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

# 3. install dependencies
pip install -r requirements.txt
```

## How to Run
```bash
streamlit run app.py
```
Other commands (run from the project folder):
```bash
python tests/test_project.py                       # 16 automated checks
python evaluation/evaluate.py --quick              # demonstration evaluation (600 profiles, ~5 s)
python evaluation/evaluate.py                      # full 4,860-profile grid (~30 s)
jupyter notebook notebooks/Workout_Recommendation_Analysis.ipynb
python scripts/generate_dataset.py                 # regenerate data/exercises.csv (optional)
python scripts/build_notebook.py                   # rebuild the notebook (optional)
```

## Launch Website
After `streamlit run app.py`, Streamlit opens the browser automatically. If it doesn't, open
**http://localhost:8501**. Stop the server with **Ctrl + C** in the terminal.
If port 8501 is busy: `streamlit run app.py --server.port 8502`.

## How to Use
1. **Quick demo:** in the sidebar choose *"2. Intermediate + Muscle Gain + Full Gym (demo)"* → **Load profile**.
2. Or fill in **1 · Welcome / Profile** (age, height, weight) and **2 · Fitness Preferences**.
3. Press **🚀 Generate My Workout**.
4. Read the **dashboard** (BMI, goal, level, duration, number of exercises, calories, time).
5. **⭐ Recommended Exercises** — ranked table, match-score chart, click an exercise for *Why recommended?*
6. **🗓️ Personalized Workout** — the full session with sets × reps, rest, time; download as CSV; **🔄 Generate Another Workout**.
7. **💡 Why These Exercises?** — score breakdown by feature and a radar chart for any recommendation.
8. **📈 Workout Statistics** and **💬 Rate This Workout**.
9. Explore **My Profile**, **History**, **Feedback**, **Dataset Explorer**, **Evaluation** and **How It Works** tabs.
10. Change **Top-N** (5/8/10) in the sidebar at any time.

## Example User
| Field | Value |
|---|---|
| Age | 21 |
| Height | 175 cm |
| Weight | 70 kg (BMI 22.9, normal range) |
| Fitness level | Intermediate |
| Goal | Muscle Gain |
| Target | Chest |
| Equipment | Full Gym |
| Duration | 60 minutes |
| Days | 5 per week |
| Preferred type | Strength |

Five sample users are included in `data/sample_users.json` (and loadable from the sidebar):
1. Beginner + Weight Loss + No Equipment
2. Intermediate + Muscle Gain + Full Gym (demo user above)
3. Advanced + Strength + Full Gym
4. Beginner + General Fitness + Home Gym
5. Intermediate + Endurance + Minimal Equipment (resistance bands)

## Example Recommendation
Actual output of the system for the example user (strict filter stage, 25 eligible exercises):

| Rank | Exercise | Match | Muscle | Equipment | Difficulty | Type |
|---|---|---|---|---|---|---|
| 1 | Smith Machine Incline Press | 90.0% | Chest | Machine | Intermediate | Strength |
| 2 | Bench Press | 88.8% | Chest | Barbell | Intermediate | Strength |
| 3 | Chest Dips | 88.2% | Chest | Full Gym (dip station) | Intermediate | Strength |
| 4 | Incline Barbell Bench Press | 88.0% | Chest | Barbell | Intermediate | Strength |
| 5 | Decline Barbell Bench Press | 86.4% | Chest | Barbell | Intermediate | Strength |

**Why Bench Press?** ✓ Matches your Muscle Gain goal · ✓ Targets your selected Chest muscle group ·
✓ Suitable for Intermediate level · ✓ Uses barbell, available with your Full Gym setup ·
✓ Matches your preferred Strength training · ✓ Fits your 60-minute session ·
✓ Movement description closely matches your preferences (TF-IDF keyword similarity).

**Generated 60-minute workout** (≈54 min incl. 6 min warm-up and 4 min cool-down, ≈309 kcal approx.):

| # | Exercise | Sets × Reps | Rest |
|---|---|---|---|
| 1 | Smith Machine Incline Press | 5 × 8-10 | 90 s |
| 2 | Bench Press | 4 × 8-10 | 90 s |
| 3 | Chest Dips | 4 × 8-10 | 90 s |
| 4 | Incline Barbell Bench Press | 4 × 8-10 | 90 s |
| 5 | Cable Fly | 3 × 10-12 | 60 s |
| 6 | Dumbbell Pullover | 3 × 10-12 | 60 s |

Pressing **Generate Another Workout** gives, for example: Chest Dips, Decline Barbell Bench Press,
Incline Dumbbell Press, Kettlebell Floor Press, Low-to-High Cable Fly, Dumbbell Fly.

## Evaluation
> **Demonstration / Test Evaluation — not real-world accuracy.**

Real Precision@K and Recall@K require relevance labels from real users or trainers, which this
project does not have. `evaluation/evaluate.py` therefore builds a **synthetic relevance set** with
transparent rules (an exercise is "relevant" if it uses available equipment, lists the user's goal,
trains the target muscle — or is conditioning work for Weight Loss / Endurance — and suits the
user's level). The labels for the 5 sample users are saved in
`evaluation/test_relevance_labels.csv`. Because these rules use the same attributes the
recommender sees, the numbers are **optimistic** and only show that the pipeline works as designed.

Metrics:
- **Precision@K** = |recommended in top K ∩ relevant| / K
- **Recall@K** = |recommended in top K ∩ relevant| / |relevant|
- **nDCG@K** — rank-aware version (higher when relevant items appear earlier)
- **Cosine similarity check** — manual formula equals scikit-learn for the demo user
- **User feedback** — ratings and 👍/👎 are collected in the app for future evaluation

Result of `python evaluation/evaluate.py` on all 4,860 level × goal × target × equipment × type
profiles (684 profiles with no synthetic relevant items skipped):

| K | System | Precision@K | Recall@K | nDCG@K |
|---|---|---|---|---|
| 5 | Content-based cosine (ours) | 0.706 | 0.545 | 0.836 |
| 5 | Filter only (no ranking) | 0.503 | 0.394 | 0.554 |
| 5 | Random (no personalisation) | 0.031 | 0.017 | 0.033 |
| 10 | Content-based cosine (ours) | 0.578 | 0.765 | 0.820 |
| 10 | Filter only (no ranking) | 0.434 | 0.558 | 0.526 |
| 10 | Random (no personalisation) | 0.031 | 0.033 | 0.036 |

Interpretation: filtering alone already removes unsuitable exercises, and cosine ranking adds a
clear improvement on top of it. (The 80% Precision@5 on the original slide was illustrative.)

## Project Structure
```
Personalized_Workout_Recommendation/
├── app.py                         # Streamlit web app
├── requirements.txt
├── README.md
├── LICENSE
├── .streamlit/config.toml         # theme
├── data/
│   ├── exercises.csv              # 298 exercises
│   ├── sample_users.json          # 5 demo profiles
│   ├── workout_history.json       # saved workouts (starts empty)
│   └── feedback.json              # saved feedback (starts empty)
├── model/
│   ├── __init__.py
│   ├── config.py                  # vocabularies, weights, rules, disclaimer
│   ├── user_profile.py            # validation, UserProfile, BMI
│   └── recommender.py             # feature encoder, filter, cosine, ranking
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py           # data cleaning
│   ├── workout_generator.py       # workout builder
│   ├── explanations.py            # "Why this exercise?"
│   ├── storage.py                 # JSON history / feedback
│   ├── timing.py                  # time estimates
│   ├── ui.py                      # CSS + HTML components
│   └── charts.py                  # Plotly charts
├── evaluation/
│   ├── __init__.py
│   ├── evaluate.py                # Precision@K, Recall@K, nDCG@K, cosine check
│   ├── test_relevance_labels.csv  # synthetic labels (demonstration)
│   └── results/                   # saved demonstration results
├── notebooks/
│   └── Workout_Recommendation_Analysis.ipynb
├── tests/
│   └── test_project.py
├── scripts/
│   ├── generate_dataset.py
│   └── build_notebook.py
├── docs/
│   └── Project_Report.md
├── assets/README.md
└── screenshots/                   # real screenshots of the running app
```

## Limitations
- The dataset is curated for education (298 exercises); real platforms use thousands with expert review.
- Calories are approximate MET-based estimates scaled by body weight, not measurements.
- Pure content-based filtering cannot learn from other users ("people like you also liked…").
- Evaluation uses synthetic relevance labels; no real user study was performed.
- Feature weights were set by reasoning, not learned from data.
- Feedback is stored but not yet used to personalise future recommendations.
- No injury, medical or progression tracking; BMI is informational only.
- History and feedback are local JSON files (single user, no login).

## Future Scope
*(Not implemented in this version.)*
- **Collaborative filtering** — learn from many users' ratings.
- **Hybrid recommendation** — combine content-based and collaborative signals.
- **Advanced user history** — long-term preferences from past workouts.
- **Progress tracking** — weights, reps and body measurements over time.
- **Mobile app** version.
- **Wearable device integration** (heart rate, steps, recovery).
- **Advanced feedback-based personalisation** — adapt feature weights from ratings.
- **AI-based workout generation** — periodised multi-week programmes.

## Conclusion
The project shows how a classic recommendation-systems technique — content-based filtering with
cosine similarity over weighted feature vectors — can produce personalised, explainable and safe
workout recommendations without any user history. The full pipeline (input → profile → features →
filtering → cosine similarity → ranking → Top-N → workout) is implemented, tested and presented in an
interactive web app, with a notebook and a transparent demonstration evaluation.

## Disclaimer
This project is developed for educational purposes. Workout recommendations are general
informational suggestions and are not medical advice or a substitute for a qualified fitness
professional. Users with injuries, medical conditions, or other health concerns should consult an
appropriate healthcare professional before exercising.
