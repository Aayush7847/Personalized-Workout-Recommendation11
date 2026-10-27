# Project Report

## Personalized Workout Recommendation System Using Machine Learning & Recommendation Systems

**Subject:** Recommendation Systems  
**Department:** Electronics & Computer Science  
**Academic Year:** 2026–27  
**Submitted by:** Aayush Deshmukh

---

## 1. Title
Personalized Workout Recommendation System Using Machine Learning & Recommendation Systems — an
AI-assisted, content-based recommender that suggests exercises and builds a complete workout from a
user's goal, fitness level, target muscle group, equipment, time and training preference.

## 2. Abstract
Generic workout plans ignore individual goals, abilities and constraints. This project implements a
content-based recommendation system in which users and exercises are represented as weighted
feature vectors in a shared 400-dimensional space built from goal, muscle group, equipment
characteristics, difficulty, exercise type, duration and a TF-IDF keyword profile. A rule-based
eligibility filter with a four-stage relaxation cascade guarantees safe and non-empty results;
eligible exercises are then ranked by cosine similarity and the Top-N are shown with explanations.
A workout generator selects a diverse subset using Maximal Marginal Relevance and prescribes sets,
repetitions and rest that fit the requested session length. The system is delivered as a Streamlit
web application over a curated dataset of 298 exercises, accompanied by a Jupyter notebook, automated
tests and a clearly labelled demonstration evaluation using Precision@K, Recall@K and nDCG@K.

## 3. Introduction
Recommendation systems help users find relevant items in large catalogues — films on Netflix, songs
on Spotify, products on Amazon. Fitness is a similar problem: there are hundreds of exercises, and the
right choice depends on what the person wants to achieve and what they can do. A beginner who wants
to lose weight at home with no equipment needs very different exercises from an advanced lifter
training legs for strength in a gym.

This project applies the content-based recommendation approach to exercise selection. It focuses on
transparency (every score can be explained), safety (hard constraints are never violated) and
practicality (the output is a ready-to-follow workout, not just a list).

## 4. Problem Statement
Given a user's age, height, weight, fitness level, primary goal, target muscle group, available
equipment, workout duration, workout days per week and preferred exercise type, recommend the most
relevant exercises from a dataset and assemble a personalised workout that:
- only uses equipment the user has,
- respects the user's fitness level,
- serves the user's goal and target muscle group,
- fits the requested duration,
- explains why each exercise was chosen.

## 5. Objectives
1. Design a web interface to collect user details and fitness preferences.
2. Build and validate a structured user profile.
3. Extract features from exercise data (categorical, numeric and text features).
4. Represent users and exercises as feature vectors in a common space.
5. Apply content-based filtering and cosine similarity to score exercises.
6. Rank exercises and provide Top-N recommendations with match percentages and explanations.
7. Generate a personalised workout with sets, repetitions, rest and time estimates.
8. Evaluate the system with standard recommender metrics and document its limitations.

## 6. Literature Review
**Recommendation approaches.** Recommender systems are broadly classified as content-based,
collaborative and hybrid (Ricci, Rokach & Shapira, 2015; Aggarwal, 2016). Collaborative filtering
predicts preferences from the behaviour of many users, while content-based systems recommend items
whose attributes resemble the user's profile (Pazzani & Billsus, 2007; Lops, de Gemmis & Semeraro,
2011). Content-based methods do not require other users' data, avoid the new-item problem and are
easy to explain, but can over-specialise and cannot exploit community preferences.

**Vector space model and TF-IDF.** Representing documents as weighted term vectors and comparing
them with cosine similarity is a classical information-retrieval technique (Salton & Buckley, 1988;
Manning, Raghavan & Schütze, 2008). TF-IDF weights terms by how frequent they are in an item and how
rare they are across the collection.

**Diversity.** Ranking purely by relevance can return near-duplicates. Maximal Marginal Relevance
(Carbonell & Goldstein, 1998) balances relevance with novelty relative to already selected items;
this project uses MMR when building the workout.

**Exercise energy expenditure and BMI.** The Compendium of Physical Activities (Ainsworth et al.,
2011) provides MET values used to approximate energy cost. BMI categories follow the World Health
Organization (WHO, 2000); BMI is used here for information only.

**Tools.** The implementation uses scikit-learn (Pedregosa et al., 2011) for TF-IDF and cosine
similarity, with Pandas, NumPy and Streamlit.

## 7. Existing System
Common existing approaches include:
- **Static plans** in books, videos or websites — one plan for everyone.
- **Fitness apps with fixed templates** — limited choices, often requiring paid subscriptions.
- **Personal trainers** — highly personalised but expensive and not always available.
- **Simple filters** (e.g. "show chest exercises") — no ranking, no consideration of multiple
  constraints at once, no explanation.

## 8. Limitations of the Existing System
- No personalisation across several factors at once (goal + level + equipment + time + type).
- Unsafe or impractical suggestions (equipment the user lacks, exercises above their level).
- No ranking or measure of how well an exercise matches the user.
- No explanation of why an exercise is suggested.
- Rigid plans that do not adapt to session length.

## 9. Proposed System
A content-based recommender that:
1. converts user inputs into a profile vector;
2. converts each exercise into a feature vector (8 weighted blocks including TF-IDF);
3. applies a hard eligibility filter with a relaxation cascade;
4. ranks eligible exercises by cosine similarity and returns the Top-N with match percentages;
5. explains each recommendation and shows a per-feature score breakdown;
6. generates a diverse, time-fitted workout and allows alternative workouts;
7. stores history and feedback locally for future analysis.

**Advantages:** personalised, explainable, safe by construction, works for brand-new users without
history, fast (a recommendation takes a few milliseconds), and runs fully offline.

## 10. System Architecture
```
User Input → Data Preprocessing → User Profile → Feature Extraction → Recommendation Engine
          → Similarity Calculation → Ranking → Top-N Recommendations → Personalized Workout
```
| Layer | Components |
|---|---|
| Presentation | Streamlit app (`app.py`, `utils/ui.py`, `utils/charts.py`) |
| Application | profile validation (`model/user_profile.py`), workout generator, explanations |
| Recommendation engine | `model/recommender.py` (FeatureEncoder, filter, cosine, ranking) |
| Data | `data/exercises.csv`, preprocessing (`utils/preprocessing.py`) |
| Persistence | `data/workout_history.json`, `data/feedback.json` (`utils/storage.py`) |

## 11. Dataset Description
`data/exercises.csv` contains 298 unique exercises generated by `scripts/generate_dataset.py` from a
hand-curated catalogue. Required columns: `exercise_id`, `exercise_name`, `muscle_group`, `equipment`,
`difficulty`, `goal`, `exercise_type`, `duration_minutes`, `calories_burned`, `sets`, `reps`,
`rest_seconds`, `instructions`, `safety_notes`. Additional columns: `mechanics`
(compound/isolation), `movement_pattern` and `tags` (keyword profile).

- Muscle groups (10): Legs 39, Back 33, Core 33, Full Body 33, Cardio 31, Chest 30, Shoulders 28, Glutes 27, Biceps 23, Triceps 21.
- Difficulty: Beginner 143, Intermediate 105, Advanced 50.
- Types: Strength 188, HIIT 38, Core 31, Mobility 23, Cardio 18.
- Equipment (11 types): No Equipment 88, Dumbbells 48, Barbell 37, Machine 30, Cable 22, Resistance Band 22, Kettlebell 17, Full Gym 13, Pull-up Bar 9, Bench 7, Treadmill 5.
- Goals: multi-label (pipe-separated, primary first).
- Calories: approximate educational estimates (MET × 3.5 × 70 kg / 200 × minutes).

## 12. Data Preprocessing
Implemented in `utils/preprocessing.py`:
1. **Schema check** — the file must exist and contain all required columns (otherwise a clear `DatasetError`).
2. **Categorical normalisation** — trimming, title-casing and alias mapping (e.g. "bodyweight" → "No Equipment").
3. **Invalid categories** — rows with missing or unknown essential categories are dropped and counted.
4. **Duplicates** — removed by exercise id and by exercise name.
5. **Numeric validation** — non-numeric, negative or out-of-range values are replaced by the median of the same exercise type.
6. **Missing values** — default reps by type; missing instructions/safety notes filled with safe defaults.
7. **Derived features** — goal list, primary goal, timed-exercise flag, kcal per minute, keyword tags.
A `CleaningReport` records every change and is shown in the app and notebook.

## 13. Feature Engineering
| Block | Encoding | Weight |
|---|---|---|
| Goal | multi-hot (primary 1.0, secondary 0.6); user: goal + related goals | 3.0 |
| Target muscle | one-hot; user: target + synergists | 3.0 |
| Equipment | 8 equipment characteristics | 2.5 |
| Fitness level | one-hot difficulty | 2.5 |
| Exercise type | one-hot; user: preferred type blended with goal-typical types | 1.5 |
| Duration | Gaussian soft bins over log minutes | 1.0 |
| Difficulty proximity | one-hot; user: graded level affinity | 1.0 |
| Keyword profile | TF-IDF of tags; user: TF-IDF of a keyword query | 1.0 |

High-weight features (goal, target muscle, equipment, level) dominate the score; medium-weight
features (type, duration, difficulty, keywords) refine the ordering. Each block is L2-normalised and
scaled by √weight before concatenation (400 dimensions in total).

## 14. Algorithm
```
1. Validate input and create UserProfile.
2. FeatureEncoder.fit(exercises) → exercise matrix X (n × 400).
3. U ← FeatureEncoder.encode_user(profile).
4. For stage in [Strict, Relaxed difficulty, Relaxed secondary, Broad]:
       eligible ← exercises passing the stage's rules (equipment always enforced)
       if |eligible| ≥ required: break
5. For each eligible exercise E: score ← (U·E) / (‖U‖‖E‖).
6. Sort by score (ties by exercise id) → Top-N.
7. Workout: MMR selection (λ = 0.72, quality floor, max 2 per movement family),
   prescription by goal and level, adjust sets to fit 90–105% of the session time.
```

## 15. Content-Based Filtering
Content-based filtering recommends items whose attributes match the user's stated preferences. Here
the "items" are exercises and their attributes are goal, muscle group, equipment, difficulty, type,
duration and keywords. Filtering happens in two layers: a **hard filter** that removes exercises the
user cannot or should not do, and a **soft ranking** by similarity. The relaxation cascade keeps the
system useful even for unusual combinations (e.g. biceps muscle gain with no equipment), and the app
tells the user when a constraint was relaxed.

## 16. Cosine Similarity
$$\cos(A,B)=\frac{A\cdot B}{\lVert A\rVert\times\lVert B\rVert}$$
Cosine similarity measures the angle between two vectors and ignores their length, so it compares
the *pattern* of features. With block-wise normalisation and weights,
$$\cos(U,E)=\frac{\sum_b w_b\cos(U_b,E_b)}{\lVert U\rVert\lVert E\rVert},$$
which lets the system report exactly how much each feature contributed to a match. For the demo user
and Bench Press: A·B = 13.768, ‖A‖ = ‖B‖ = 3.937, cosine = 0.888 → 88.8% match (the manual value equals
scikit-learn's).

## 17. Implementation
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI: form, dashboard, recommendations, workout, explanations, statistics, history, feedback, dataset explorer, evaluation, how-it-works |
| `model/config.py` | vocabularies, feature weights, eligibility rules, prescription templates, disclaimer |
| `model/user_profile.py` | input validation and BMI |
| `model/recommender.py` | feature encoding, filtering, cosine similarity, ranking |
| `utils/workout_generator.py` | workout assembly and time fitting |
| `utils/explanations.py` | dynamic reasons and score breakdown |
| `utils/storage.py` | atomic JSON writes with corruption backup |
| `evaluation/evaluate.py` | metrics and baselines |
| `notebooks/…ipynb` | step-by-step analysis |
| `tests/test_project.py` | 16 automated checks |

Implementation notes: the exercise matrix is built once and cached (`st.cache_resource`); scoring
is a single vectorised matrix operation; history writes are atomic (temporary file + rename); every
error path shows a friendly message instead of a stack trace.

## 18. Results
For the demo user (21 years, 175 cm, 70 kg, Intermediate, Muscle Gain, Chest, Full Gym, 60 min,
5 days, Strength), 25 exercises pass the strict filter. Top-5: Smith Machine Incline Press 90.0%,
Bench Press 88.8%, Chest Dips 88.2%, Incline Barbell Bench Press 88.0%, Decline Barbell Bench Press
86.4%. The generated workout has 6 exercises (5×8-10 to 3×10-12) and takes about 54 minutes including
warm-up and cool-down, ≈309 kcal (approximate). An alternative workout shares only one exercise with
the first. All five sample users receive valid workouts, and an exhaustive test over all 4,860
categorical profile combinations confirms that results are never empty and never require equipment
the user lacks.

Screenshots of the running application are in the `screenshots/` folder.

## 19. Evaluation
**Demonstration / Test Evaluation — not real-world accuracy.** No real users or expert labels were
available, so relevance was defined by transparent synthetic rules (equipment available, goal listed,
target muscle trained or conditioning work for Weight Loss/Endurance, suitable difficulty). Because
these rules use the same attributes as the recommender, the results are optimistic.

| K | System | Precision@K | Recall@K | nDCG@K |
|---|---|---|---|---|
| 5 | Content-based cosine (ours) | 0.706 | 0.545 | 0.836 |
| 5 | Filter only | 0.503 | 0.394 | 0.554 |
| 5 | Random | 0.031 | 0.017 | 0.033 |
| 10 | Content-based cosine (ours) | 0.578 | 0.765 | 0.820 |
| 10 | Filter only | 0.434 | 0.558 | 0.526 |
| 10 | Random | 0.031 | 0.033 | 0.036 |

(4,176 profiles evaluated; 684 without any synthetic relevant item skipped.) The comparison shows
that both stages matter: filtering removes unsuitable exercises, and cosine ranking orders the
remainder much better than chance. A proper evaluation would use expert-labelled relevance or a
user study with the in-app feedback.

## 20. Advantages
- Personalised on ten inputs simultaneously.
- Works for new users immediately (no history required).
- Explainable: per-feature contributions sum exactly to the score.
- Safe constraints (equipment, level) are enforced before ranking.
- Never returns an empty result.
- Fast and fully offline; simple to extend (new exercises are just new CSV rows).

## 21. Limitations
- Curated educational dataset; not reviewed by certified trainers.
- Approximate calorie estimates.
- Feature weights chosen by design reasoning, not learned.
- No collaborative signal; feedback is collected but not yet used for learning.
- Synthetic evaluation labels; no user study.
- No injury handling, progression or periodisation; single-user local storage.

## 22. Future Scope
(Not implemented in this version.)
- Collaborative filtering and hybrid recommendation.
- Advanced user history and progress tracking.
- Mobile application and wearable-device integration.
- Feedback-driven personalisation (learning feature weights from ratings).
- AI-based multi-week workout programme generation.

## 23. Conclusion
The project demonstrates a complete content-based recommendation pipeline — user input, profile
building, feature extraction with TF-IDF, content-based filtering, cosine similarity, ranking, Top-N
recommendations and workout generation — in an interactive, explainable web application. It meets
the objectives of personalising workouts to the user's goal, level, equipment and time while keeping
recommendations safe and transparent, and it provides a solid base for future hybrid and
feedback-driven extensions.

---

## References
1. Pazzani, M. J., & Billsus, D. (2007). Content-Based Recommendation Systems. In *The Adaptive Web* (LNCS 4321, pp. 325–341). Springer.
2. Lops, P., de Gemmis, M., & Semeraro, G. (2011). Content-based Recommender Systems: State of the Art and Trends. In *Recommender Systems Handbook* (pp. 73–105). Springer.
3. Ricci, F., Rokach, L., & Shapira, B. (Eds.). (2015). *Recommender Systems Handbook* (2nd ed.). Springer.
4. Aggarwal, C. C. (2016). *Recommender Systems: The Textbook*. Springer.
5. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513–523.
6. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
7. Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. *Proceedings of SIGIR '98*, 335–336.
8. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
9. Ainsworth, B. E., et al. (2011). 2011 Compendium of Physical Activities: a second update of codes and MET values. *Medicine & Science in Sports & Exercise*, 43(8), 1575–1581.
10. World Health Organization. (2000). *Obesity: Preventing and Managing the Global Epidemic* (WHO Technical Report Series 894). Geneva: WHO.

---

> ⚠️ **Disclaimer:** This project is developed for educational purposes. Workout recommendations are general informational suggestions and are not medical advice or a substitute for a qualified fitness professional. Users with injuries, medical conditions, or other health concerns should consult an appropriate healthcare professional before exercising.
