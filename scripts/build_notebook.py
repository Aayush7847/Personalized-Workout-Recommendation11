"""
Builds notebooks/Workout_Recommendation_Analysis.ipynb.

The notebook imports the SAME modules as the Streamlit app, so every result
in it is produced by the real recommender (no duplicated logic).

    python scripts/build_notebook.py          # writes the .ipynb
    jupyter nbconvert --to notebook --execute --inplace notebooks/Workout_Recommendation_Analysis.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "Workout_Recommendation_Analysis.ipynb"

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Personalized Workout Recommendation System — Analysis Notebook

**Subject:** Recommendation Systems · **Technique:** Content-Based Filtering + Cosine Similarity

This notebook walks through the complete pipeline used by the Streamlit app:

`User Input → User Profile → Feature Extraction → Content-Based Filtering → Feature Vectors → Cosine Similarity → Ranking → Top-N → Personalized Workout`

It imports the **same modules** as `app.py` (`model/`, `utils/`, `evaluation/`), so the numbers here are exactly what the web app produces.

> ⚠️ Educational project. Recommendations are general informational suggestions, not medical advice.
""")

md("## 1. Import libraries")
code("""
import sys, json, warnings
from pathlib import Path

# Make the project root importable whether the notebook runs from notebooks/ or the root
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model.config import FEATURE_WEIGHTS, BLOCK_LABELS, DISCLAIMER
from model.user_profile import UserProfile, build_profile
from model.recommender import WorkoutRecommender, FeatureEncoder, BLOCK_ORDER, FILTER_STAGES
from utils.preprocessing import load_raw_dataset, clean_exercise_data
from utils.workout_generator import recommend_and_generate, generate_workout
from utils.explanations import explain_exercise, score_breakdown
from evaluation.evaluate import (precision_at_k, recall_at_k, cosine_similarity_manual,
                                 evaluate_sample_users, evaluate_grid, cosine_verification)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)
plt.rcParams.update({"figure.dpi": 100, "axes.spines.top": False, "axes.spines.right": False})
print("Project root:", ROOT)
""")

md("## 2. Load the dataset")
code("""
DATA = ROOT / "data" / "exercises.csv"
raw = load_raw_dataset(DATA)
print(f"Loaded {len(raw)} rows × {raw.shape[1]} columns from {DATA.relative_to(ROOT)}")
raw.head()
""")

md("## 3. Dataset overview")
code("""
raw.info()
""")
code("""
overview = pd.DataFrame({
    "Unique values": raw.nunique(),
    "Example": raw.iloc[0],
})
overview
""")
code("""
raw[["duration_minutes", "calories_burned", "sets", "rest_seconds"]].describe().round(1)
""")
md("Calories are **approximate educational values** (MET-based estimates for a ~70 kg person), not medically accurate measurements.")

md("""
## 4. Data cleaning

`utils.preprocessing.clean_exercise_data` performs every step below. To *demonstrate* it,
we create a deliberately corrupted copy of the data (the real file is left untouched):
a duplicate row, a missing muscle group, an invalid negative duration, a text value in a numeric
column, inconsistent capitalisation and a missing instruction.
""")
code("""
dirty = raw.copy()
dirty = pd.concat([dirty, dirty.iloc[[0]]], ignore_index=True)        # duplicate exercise
dirty.loc[1, "muscle_group"] = np.nan                                  # missing category
dirty.loc[2, "duration_minutes"] = -5                                  # invalid numeric
dirty["calories_burned"] = dirty["calories_burned"].astype(object)
dirty.loc[3, "calories_burned"] = "lots"                               # non-numeric value
dirty.loc[4, "difficulty"] = "  beginner "                             # inconsistent text
dirty.loc[5, "instructions"] = np.nan                                  # missing text

cleaned_demo, report = clean_exercise_data(dirty)
pd.Series(report.as_dict(), name="Cleaning report").to_frame()
""")

md("## 5. Missing-value analysis")
code("""
missing = pd.DataFrame({"Missing (corrupted copy)": dirty.isna().sum(),
                        "Missing (after cleaning)": cleaned_demo[dirty.columns].isna().sum()})
missing[missing.sum(axis=1) > 0]
""")
code("""
# The real dataset, cleaned exactly as the app does
df, real_report = clean_exercise_data(raw)
print(real_report.as_dict())
print("Missing values in cleaned data:", int(df[raw.columns].isna().sum().sum()))
""")

md("## 6. Exploratory data analysis")
code("""
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
df["muscle_group"].value_counts().sort_values().plot.barh(ax=axes[0, 0], color="#2347D6")
axes[0, 0].set_title("Exercises by muscle group")
df["difficulty"].value_counts().reindex(["Beginner", "Intermediate", "Advanced"]).plot.bar(
    ax=axes[0, 1], color=["#1F9D6B", "#2347D6", "#FF6B2C"], rot=0)
axes[0, 1].set_title("Exercises by difficulty")
df["equipment"].value_counts().plot.bar(ax=axes[1, 0], color="#FF6B2C", rot=35)
axes[1, 0].set_title("Exercises by equipment")
df["goals_list"].explode().value_counts().plot.bar(ax=axes[1, 1], color="#7A4DD8", rot=20)
axes[1, 1].set_title("Exercises by goal (multi-label)")
plt.tight_layout(); plt.show()
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for t, g in df.groupby("exercise_type"):
    axes[0].hist(g["duration_minutes"], bins=15, alpha=.6, label=t)
axes[0].set_title("Exercise duration distribution"); axes[0].set_xlabel("minutes"); axes[0].legend()
axes[1].scatter(df["duration_minutes"], df["calories_burned"], s=14, alpha=.6, c="#2347D6")
axes[1].set_title("Duration vs approximate calories"); axes[1].set_xlabel("minutes"); axes[1].set_ylabel("kcal (approx.)")
plt.tight_layout(); plt.show()
""")
code("""
pd.crosstab(df["muscle_group"], df["difficulty"])[["Beginner", "Intermediate", "Advanced"]]
""")

md("""
## 7. Feature engineering

Each exercise is represented by **8 feature blocks**:

| Block | Encoding | Weight |
|---|---|---|
| Goal | multi-hot (primary goal 1.0, secondary 0.6) | 3.0 |
| Target muscle | one-hot over muscle groups | 3.0 |
| Equipment | 8 equipment characteristics (bodyweight, free weight, machine, elastic, …) | 2.5 |
| Fitness level | one-hot difficulty | 2.5 |
| Exercise type | one-hot type | 1.5 |
| Duration | soft Gaussian bins over log(minutes) | 1.0 |
| Difficulty proximity | one-hot difficulty (user side: graded affinity) | 1.0 |
| Keyword profile | **TF-IDF** of the exercise's keyword tags | 1.0 |

Each block is L2-normalised and multiplied by √weight, then all blocks are concatenated.
""")
code("""
pd.DataFrame({"Feature block": [BLOCK_LABELS[b] for b in BLOCK_ORDER],
              "Weight": [FEATURE_WEIGHTS[b] for b in BLOCK_ORDER]})
""")

md("## 8. Encoding (fitting the feature encoder)")
code("""
rec = WorkoutRecommender(df)          # fits FeatureEncoder on the cleaned dataset
enc = rec.encoder
print("Exercise matrix shape:", rec.X.shape)
pd.DataFrame({"Block": [BLOCK_LABELS[b] for b in BLOCK_ORDER],
              "Dimensions": [len(enc.vocab[b]) for b in BLOCK_ORDER]})
""")
code("""
# TF-IDF vocabulary sample and one exercise's keyword document
print("TF-IDF terms:", len(enc.vocab["text"]), "→", enc.vocab["text"][:20])
print("Bench Press tags:", df.loc[df.exercise_name == "Bench Press", "tags"].iloc[0])
""")

md("## 9. User profile (demo user)")
code("""
demo_input = {"age": 21, "height_cm": 175, "weight_kg": 70, "fitness_level": "Intermediate",
              "goal": "Muscle Gain", "target_muscle": "Chest", "equipment": "Full Gym",
              "duration": 60, "days_per_week": 5, "exercise_type": "Strength"}
profile, errors = build_profile(demo_input)
print("Validation errors:", errors)
print(profile)
print(f"BMI = {profile.bmi} ({profile.bmi_category}) — informational only")
""")
code("""
# Input validation example
print(build_profile({**demo_input, "age": 7, "goal": None})[1])
""")

md("## 10. User feature vector")
code("""
user_vec = enc.encode_user(profile).ravel()
vec_df = pd.DataFrame({"feature": enc.feature_names(), "value": user_vec})
nonzero = vec_df[vec_df.value.abs() > 1e-9]
print(f"{len(nonzero)} of {len(user_vec)} dimensions are non-zero; ||U|| = {np.linalg.norm(user_vec):.4f}")
nonzero.head(30).round(4)
""")

md("""
## 11. Cosine similarity

$$\\cos(A,B)=\\frac{A\\cdot B}{\\lVert A\\rVert\\,\\lVert B\\rVert}$$

We compute it manually for one exercise and compare with scikit-learn (used by the engine).
""")
code("""
from sklearn.metrics.pairwise import cosine_similarity
row = df.index[df.exercise_name == "Bench Press"][0]
A, B = user_vec, rec.X[row]
manual = A @ B / (np.linalg.norm(A) * np.linalg.norm(B))
sk = cosine_similarity(A.reshape(1, -1), B.reshape(1, -1))[0, 0]
print(f"A·B = {A @ B:.4f}, ||A|| = {np.linalg.norm(A):.4f}, ||B|| = {np.linalg.norm(B):.4f}")
print(f"manual cosine = {manual:.6f} | scikit-learn = {sk:.6f} | equal: {np.isclose(manual, sk)}")
""")
code("""
cosine_verification(rec).round(4)
""")

md("## 12. Content-based filtering + ranking")
code("""
result = rec.recommend(profile, top_n=10)
print("Filter stage:", result.filter_stage, "| eligible exercises:", result.n_eligible, "of", len(df))
print("Relaxation cascade:", FILTER_STAGES)
result.ranked[["rank", "exercise_name", "match_percent", "muscle_group", "equipment",
               "difficulty", "exercise_type"]].head(15)
""")
code("""
# Every score decomposes exactly into per-feature contributions
top = result.top_n
contrib = top[[f"contrib_{b}" for b in BLOCK_ORDER]].sum(axis=1)
print("max |sum(contributions) - cosine| =", float(np.abs(contrib - top["similarity"]).max()))
score_breakdown(top.iloc[0])
""")

md("## 13. Top-N recommendations with explanations")
code("""
for _, r in top.head(5).iterrows():
    print(f"#{r['rank']} {r['exercise_name']} — {r['match_percent']:.1f}% match")
    for reason in explain_exercise(r, profile):
        print("    ", reason)
""")

md("## 14. Workout generation")
code("""
result, workout = recommend_and_generate(rec, profile, top_n=5)
wdf = workout.to_frame()[["order", "exercise_name", "sets", "reps", "rest_seconds",
                          "est_minutes", "calories", "match_percent"]]
print(f"Warm-up {workout.warmup_minutes} min + exercises {workout.exercise_minutes} min + "
      f"cool-down {workout.cooldown_minutes} min = {workout.total_minutes} min "
      f"(requested {profile.duration}); ≈{workout.total_calories(profile.weight_kg)} kcal (approx.)")
wdf
""")
code("""
# "Generate another workout": same constraints, different exercises where possible
alt = generate_workout(result, rec, variation=1, exclude_ids=workout.exercise_ids)
print("Original :", [i.exercise_name for i in workout.items])
print("Alternate:", [i.exercise_name for i in alt.items])
print("Shared exercises:", len(set(workout.exercise_ids) & set(alt.exercise_ids)))
""")
code("""
# All five sample users from data/sample_users.json
for d in json.loads((ROOT / "data" / "sample_users.json").read_text()):
    label = d.pop("label"); p = UserProfile.from_dict(d)
    r, w = recommend_and_generate(rec, p, top_n=5)
    print(f"{label}\\n   stage={r.filter_stage}, eligible={r.n_eligible}, "
          f"{len(w.items)} exercises, {w.total_minutes} / {p.duration} min")
    print("   ", ", ".join(i.exercise_name for i in w.items))
""")

md("""
## 15. Evaluation — DEMONSTRATION / TEST ONLY

Real Precision@K / Recall@K need relevance labels from real users or trainers, which this
project does not have. `evaluation/evaluate.py` therefore builds a **clearly labelled synthetic
relevance set** from transparent rules. These numbers show *how* the metrics work; they are
**not** real-world accuracy and are optimistic by construction.
""")
code("""
# Tiny worked example of the metric definitions
recommended = ["A", "B", "C", "D", "E"]; relevant = {"A", "C", "F", "G"}
print("Precision@5 =", precision_at_k(recommended, relevant, 5), "| Recall@5 =", recall_at_k(recommended, relevant, 5))
""")
code("""
users_eval = evaluate_sample_users(rec, k=5)
users_eval[["Profile", "n_relevant", "ours_precision", "ours_recall", "filter_precision", "random_precision"]].round(3)
""")
code("""
grid = evaluate_grid(rec, ks=(5, 10), max_profiles=300)
grid.round(3)
""")

md("## 16. Visualizations")
code("""
fig, ax = plt.subplots(figsize=(9, 4))
t = result.top_n.iloc[::-1]
ax.barh(t["exercise_name"], t["match_percent"], color="#2347D6")
for y, v in enumerate(t["match_percent"]):
    ax.text(v + 0.5, y, f"{v:.1f}%", va="center")
ax.set_xlim(0, 105); ax.set_xlabel("Match score (%)"); ax.set_title("Top-5 recommendations: exercise vs match score")
plt.tight_layout(); plt.show()
""")
code("""
fig, ax = plt.subplots(figsize=(10, 4))
t = result.top_n.iloc[::-1]
left = np.zeros(len(t))
for b in BLOCK_ORDER:
    vals = 100 * t[f"contrib_{b}"].to_numpy()
    ax.barh(t["exercise_name"], vals, left=left, label=BLOCK_LABELS[b]); left += vals
ax.set_xlabel("Contribution to match score (points)"); ax.set_title("Score breakdown by feature block")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.tight_layout(); plt.show()
""")
code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(result.ranked["match_percent"], bins=20, color="#FF6B2C")
axes[0].set_title("Match scores of all eligible exercises"); axes[0].set_xlabel("%")
pivot = grid.pivot(index="K", columns="System", values="Precision@K")
pivot.plot.bar(ax=axes[1], rot=0); axes[1].set_ylim(0, 1)
axes[1].set_title("Precision@K (demonstration labels)")
plt.tight_layout(); plt.show()
""")

md(f"""
## 17. Conclusion

* The recommender represents users and exercises in the **same weighted feature space** and ranks
  eligible exercises by **cosine similarity** — a transparent content-based approach that needs no
  user history (no cold-start problem for new users).
* A hard filter with a relaxation cascade guarantees safe, non-empty results.
* Every score decomposes into per-feature contributions, so each recommendation is explainable.
* The workout generator turns the ranking into a time-fitted, diverse session.
* Evaluation here is a **demonstration** with synthetic labels; real validation would need user studies.

> ⚠️ This project is developed for educational purposes. Workout recommendations are general informational suggestions and are not medical advice or a substitute for a qualified fitness professional.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
OUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, OUT)
print(f"Wrote {OUT.relative_to(ROOT)} ({len(cells)} cells)")
