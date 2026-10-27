"""
PERSONALIZED WORKOUT - AI-Assisted Recommendation System
========================================================

Streamlit front-end for the content-based workout recommender.

    pip install -r requirements.txt
    streamlit run app.py

Pipeline (same engine as the notebook and the evaluation script):

    User Input -> User Profile -> Feature Extraction -> Content-Based Filtering
    -> Feature Vectors -> Cosine Similarity -> Ranking -> Top-N
    -> Personalized Workout

This file only handles the user interface. All recommendation logic lives in
model/recommender.py and utils/workout_generator.py.
"""
from __future__ import annotations

import inspect
import json
from html import escape
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.config import (BLOCK_LABELS, DISCLAIMER, DURATIONS,  # noqa: E402
                          EXERCISE_COUNT_RANGE, FITNESS_LEVELS, GOALS,
                          TARGET_MUSCLES, TOP_N_OPTIONS, USER_EQUIPMENT,
                          USER_EXERCISE_TYPES, WORKOUT_DAYS)
from model.recommender import (BLOCK_ORDER, FILTER_STAGES,  # noqa: E402
                               RecommendationResult, WorkoutRecommender)
from model.user_profile import (AGE_RANGE, HEIGHT_RANGE_CM,  # noqa: E402
                                WEIGHT_RANGE_KG, UserProfile, build_profile)
from utils import charts, ui  # noqa: E402
from utils.explanations import explain_exercise, score_breakdown  # noqa: E402
from utils.preprocessing import (DEFAULT_DATASET_PATH, CleaningReport,  # noqa: E402
                                 DatasetError, load_exercise_data)
from utils.storage import (feedback_store, history_store,  # noqa: E402
                           workout_to_history_record)
from utils.workout_generator import (MAX_SETS_BY_LEVEL, Workout,  # noqa: E402
                                     generate_workout, recommend_and_generate)

SAMPLE_USERS_PATH = ROOT / "data" / "sample_users.json"
EVAL_RESULTS_DIR = ROOT / "evaluation" / "results"

st.set_page_config(page_title="Personalized Workout · Recommender", page_icon="🏋️",
                   layout="wide", initial_sidebar_state="expanded")


# ============================================================================
# Small compatibility helper (Streamlit renamed use_container_width -> width)
# ============================================================================
def wide(fn) -> dict:
    """Keyword arguments that make an element full-width on any Streamlit >= 1.36."""
    param = inspect.signature(fn).parameters.get("width")
    if param is not None and isinstance(param.default, str):
        return {"width": "stretch"}
    return {"use_container_width": True}


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def plot(fig, key: str | None = None) -> None:
    st.plotly_chart(fig, key=key, config={"displayModeBar": False}, **wide(st.plotly_chart))


# ============================================================================
# Data / engine loading (cached)
# ============================================================================
@st.cache_resource(show_spinner="Loading exercise dataset and building feature vectors…")
def _load_engine(path: str, mtime: float) -> tuple[WorkoutRecommender, CleaningReport]:
    df, report = load_exercise_data(path)
    return WorkoutRecommender(df), report


def get_engine() -> tuple[WorkoutRecommender, CleaningReport]:
    """Load the recommender, stopping the app with a friendly message on failure."""
    path = Path(DEFAULT_DATASET_PATH)
    if not path.exists():
        st.error(f"📂 The exercise dataset was not found at `{path.relative_to(ROOT)}`. "
                 "Please make sure the `data/` folder from the project ZIP is present, "
                 "or regenerate it with `python scripts/generate_dataset.py`.")
        st.stop()
    try:
        return _load_engine(str(path), path.stat().st_mtime)
    except DatasetError as exc:
        st.error(f"⚠️ The exercise dataset could not be used: {exc}")
        st.stop()
    except Exception as exc:  # pragma: no cover - last-resort guard
        st.error(f"⚠️ Unexpected error while loading the dataset: {exc}")
        st.stop()


@st.cache_data(show_spinner=False)
def load_sample_users() -> dict[str, dict]:
    try:
        records = json.loads(SAMPLE_USERS_PATH.read_text(encoding="utf-8"))
        return {r["label"]: r for r in records if isinstance(r, dict) and "label" in r}
    except (OSError, json.JSONDecodeError, TypeError):
        return {}


ENGINE, CLEANING_REPORT = get_engine()
SAMPLES = load_sample_users()

# ============================================================================
# Session state
# ============================================================================
FORM_KEYS = {"age": "f_age", "height_cm": "f_height", "weight_kg": "f_weight",
             "fitness_level": "f_level", "goal": "f_goal", "target_muscle": "f_target",
             "equipment": "f_equipment", "duration": "f_duration",
             "days_per_week": "f_days", "exercise_type": "f_type"}
NO_SAMPLE = "— choose a sample profile —"

for _key in FORM_KEYS.values():
    st.session_state.setdefault(_key, None)
for _key, _default in {"profile": None, "result": None, "workout": None, "variation": 0,
                       "form_errors": [], "flash": None, "feedback_done": False,
                       "history_id": None, "top_n": TOP_N_OPTIONS[0],
                       "sample_choice": NO_SAMPLE}.items():
    st.session_state.setdefault(_key, _default)


def _numeric_or_none(value, kind=float):
    try:
        return kind(value)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- callbacks
def cb_load_sample() -> None:
    sample = SAMPLES.get(st.session_state.sample_choice)
    if not sample:
        st.session_state.flash = ("warning", "Please choose a sample profile first.")
        return
    for field, key in FORM_KEYS.items():
        value = sample.get(field)
        if field == "age":
            value = _numeric_or_none(value, int)
        elif field in ("height_cm", "weight_kg"):
            value = _numeric_or_none(value, float)
        st.session_state[key] = value
    st.session_state.form_errors = []
    st.session_state.flash = ("success", f"Loaded sample profile: {sample['label']}. "
                                         "Press **Generate My Workout**.")


def cb_clear_form() -> None:
    for key in FORM_KEYS.values():
        st.session_state[key] = None
    for key in ("profile", "result", "workout", "history_id"):
        st.session_state[key] = None
    st.session_state.form_errors = []
    st.session_state.variation = 0
    st.session_state.feedback_done = False
    st.session_state.sample_choice = NO_SAMPLE


def _save_history(profile: UserProfile, workout: Workout, stage: str) -> None:
    try:
        record = history_store().append(workout_to_history_record(profile, workout, stage))
        st.session_state.history_id = record["id"]
    except OSError as exc:
        st.session_state.history_id = None
        st.session_state.flash = ("warning", f"Workout generated, but history could not be "
                                             f"saved ({exc}).")


def cb_generate() -> None:
    """Validate the form, run the recommender and build the workout."""
    raw = {field: st.session_state.get(key) for field, key in FORM_KEYS.items()}
    profile, errors = build_profile(raw)
    st.session_state.form_errors = errors
    if errors:
        return
    try:
        result, workout = recommend_and_generate(ENGINE, profile, top_n=max(TOP_N_OPTIONS))
    except Exception as exc:  # pragma: no cover - defensive
        st.session_state.form_errors = [f"Could not generate recommendations: {exc}"]
        return
    if result.ranked.empty or not workout.items:
        st.session_state.form_errors = [
            "No exercises match your equipment selection. Try a different equipment option."]
        return
    st.session_state.update(profile=profile, result=result, workout=workout, variation=0,
                            feedback_done=False)
    _save_history(profile, workout, result.filter_stage)


def cb_regenerate() -> None:
    """Build an alternative workout for the same profile (same constraints)."""
    result: RecommendationResult | None = st.session_state.result
    workout: Workout | None = st.session_state.workout
    if result is None or workout is None:
        return
    variation = st.session_state.variation + 1
    new = generate_workout(result, ENGINE, variation=variation, exclude_ids=workout.exercise_ids)
    st.session_state.update(workout=new, variation=variation, feedback_done=False)
    _save_history(result.profile, new, result.filter_stage)


# ============================================================================
# Shared render helpers
# ============================================================================
def render_reasons(row: pd.Series, profile: UserProfile) -> None:
    html("".join(ui.reason_html(r.status, r.icon, r.text) for r in explain_exercise(row, profile)))


def exercise_chips(row) -> list[str]:
    return [ui.chip(row["muscle_group"], "blue"), ui.chip(row["equipment"]),
            ui.chip(row["difficulty"], "green"), ui.chip(row["exercise_type"], "orange")]


def top_frame(result: RecommendationResult, top_n: int) -> pd.DataFrame:
    return result.ranked.head(top_n)


# ============================================================================
# Layout: hero + sidebar
# ============================================================================
html(ui.CSS)
html(ui.hero_html())

with st.sidebar:
    st.markdown("### ⚡ Quick start")
    st.caption("Load one of the 5 sample users, or fill in the form yourself.")
    st.selectbox("Sample profile", [NO_SAMPLE, *SAMPLES.keys()], key="sample_choice")
    c1, c2 = st.columns(2)
    c1.button("Load profile", on_click=cb_load_sample, type="primary", **wide(st.button))
    c2.button("Clear form", on_click=cb_clear_form, **wide(st.button))

    st.markdown("### 🎯 Recommendations")
    st.radio("Top-N exercises to show", TOP_N_OPTIONS, horizontal=True, key="top_n",
             help="How many ranked exercises appear in the recommendation list.")

    st.markdown("### 🗂️ Dataset")
    st.caption(f"✓ **{len(ENGINE.df)} exercises** loaded and encoded into "
               f"**{ENGINE.X.shape[1]}-dimensional** feature vectors.")
    _dropped = CLEANING_REPORT.rows_in - CLEANING_REPORT.rows_out
    if _dropped:
        st.caption(f"Preprocessing removed {_dropped} invalid/duplicate row(s).")
    st.markdown("---")
    html(ui.disclaimer_html(DISCLAIMER))

if st.session_state.flash:
    kind, message = st.session_state.flash
    getattr(st, kind)(message)
    st.session_state.flash = None

(tab_plan, tab_profile, tab_history, tab_feedback, tab_data, tab_eval,
 tab_how) = st.tabs(["🏋️ Plan My Workout", "👤 My Profile", "🕘 History", "💬 Feedback",
                     "📊 Dataset Explorer", "🧪 Evaluation", "ℹ️ How It Works"])


# ============================================================================
# TAB 1 - Plan my workout
# ============================================================================
def render_form() -> None:
    html(ui.section_html("1", "Welcome / Profile",
                         "Tell us about yourself — used for BMI and calorie estimates."))
    with st.form("profile_form", border=True):
        c1, c2, c3 = st.columns(3)
        c1.number_input(f"Age (years, {AGE_RANGE[0]}–{AGE_RANGE[1]})", key=FORM_KEYS["age"],
                        step=1, placeholder="e.g. 21")
        c2.number_input(f"Height (cm, {HEIGHT_RANGE_CM[0]:g}–{HEIGHT_RANGE_CM[1]:g})",
                        key=FORM_KEYS["height_cm"], step=1.0, format="%.1f", placeholder="e.g. 175")
        c3.number_input(f"Weight (kg, {WEIGHT_RANGE_KG[0]:g}–{WEIGHT_RANGE_KG[1]:g})",
                        key=FORM_KEYS["weight_kg"], step=0.5, format="%.1f", placeholder="e.g. 70")

        html(ui.section_html("2", "Fitness Preferences",
                             "These drive the content-based recommendation."))
        c1, c2, c3 = st.columns(3)
        c1.selectbox("Fitness level", FITNESS_LEVELS, key=FORM_KEYS["fitness_level"],
                     placeholder="Select level…")
        c2.selectbox("Primary goal", GOALS, key=FORM_KEYS["goal"], placeholder="Select goal…")
        c3.selectbox("Target muscle group", TARGET_MUSCLES, key=FORM_KEYS["target_muscle"],
                     placeholder="Select target…")
        c1, c2, c3, c4 = st.columns(4)
        c1.selectbox("Available equipment", USER_EQUIPMENT, key=FORM_KEYS["equipment"],
                     placeholder="Select equipment…")
        c2.selectbox("Workout duration", DURATIONS, key=FORM_KEYS["duration"],
                     format_func=lambda m: f"{m} min", placeholder="Select duration…")
        c3.selectbox("Workout days per week", WORKOUT_DAYS, key=FORM_KEYS["days_per_week"],
                     format_func=lambda d: f"{d} days", placeholder="Select days…")
        c4.selectbox("Preferred exercise type", USER_EXERCISE_TYPES, key=FORM_KEYS["exercise_type"],
                     placeholder="Select type…")

        html(ui.section_html("3", "Generate Workout"))
        st.form_submit_button("🚀 Generate My Workout", type="primary", on_click=cb_generate,
                              **wide(st.form_submit_button))

    if st.session_state.form_errors:
        st.error("Please fix the following before generating:\n\n" +
                 "\n".join(f"- {e}" for e in st.session_state.form_errors))


def render_dashboard(profile: UserProfile, result: RecommendationResult, workout: Workout) -> None:
    lo, hi = EXERCISE_COUNT_RANGE[profile.duration]
    kcal = workout.total_calories(profile.weight_kg)
    cards = [
        ui.metric_card("BMI", f"{profile.bmi:.1f}", f"{profile.bmi_category} · informational",
                       ui.MUTED),
        ui.metric_card("Goal", profile.goal, f"Target: {profile.target_muscle}", ui.ACCENT),
        ui.metric_card("Fitness level", profile.fitness_level, profile.equipment),
        ui.metric_card("Duration", f"{profile.duration} min", f"{profile.days_per_week} days / week"),
        ui.metric_card("Exercises", str(len(workout.items)), f"planned range {lo}–{hi}", ui.POSITIVE),
        ui.metric_card("Est. calories", f"~{kcal} kcal", "approximate, educational", ui.ACCENT),
        ui.metric_card("Est. time", f"{workout.total_minutes:.0f} min",
                       "incl. warm-up & cool-down", ui.POSITIVE),
    ]
    html(ui.metric_grid(cards))
    share = workout.total_minutes / profile.duration
    st.progress(min(share, 1.0),
                text=f"Session time used: {workout.total_minutes:.0f} of {profile.duration} min "
                     f"({share:.0%}) · {result.n_eligible} exercises passed the filter "
                     f"({result.filter_stage.lower()})")
    st.caption("BMI = weight (kg) / height (m)². It is shown for information only, is not a "
               "medical assessment and is **not** used to choose exercises.")


def render_recommendations(profile: UserProfile, result: RecommendationResult, top_n: int) -> None:
    top = top_frame(result, top_n)
    st.markdown(f"**Top {len(top)} of {result.n_eligible} eligible exercises**, ranked by cosine "
                "similarity between your profile vector and each exercise vector.")
    table = top[["rank", "exercise_name", "match_percent", "muscle_group", "equipment",
                 "difficulty", "exercise_type"]].rename(columns={
                     "rank": "Rank", "exercise_name": "Exercise", "match_percent": "Match Score",
                     "muscle_group": "Muscle Group", "equipment": "Equipment",
                     "difficulty": "Difficulty", "exercise_type": "Exercise Type"})
    st.dataframe(table, hide_index=True, column_config={
        "Rank": st.column_config.NumberColumn(width="small"),
        "Match Score": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100),
    }, **wide(st.dataframe))
    plot(charts.match_score_bar(top), key="match_bar")

    st.markdown("#### Why recommended? — click an exercise")
    for _, row in top.iterrows():
        with st.expander(f"#{int(row['rank'])}  {row['exercise_name']} — "
                         f"{row['match_percent']:.1f}% match"):
            left, right = st.columns([3, 2])
            with left:
                html(f'<div class="pw-score">{row["match_percent"]:.1f}% MATCH</div>')
                render_reasons(row, profile)
            with right:
                html("".join(exercise_chips(row)))
                st.markdown(f"**How to:** {row['instructions']}")
                st.markdown(f"**Safety:** {row['safety_notes']}")


def render_workout(profile: UserProfile, result: RecommendationResult, workout: Workout) -> None:
    head, action = st.columns([3, 1])
    with head:
        html(ui.chip(f"Goal: {profile.goal}", "orange") + ui.chip(f"Target: {profile.target_muscle}", "blue")
             + ui.chip(f"Level: {profile.fitness_level}", "green")
             + ui.chip(f"Duration: {profile.duration} min") + ui.chip(f"Equipment: {profile.equipment}")
             + (ui.chip(f"Alternative #{workout.variation}") if workout.variation else ""))
    with action:
        st.button("🔄 Generate Another Workout", on_click=cb_regenerate, type="primary",
                  help="Same goal, target, equipment, level and duration — different high-ranked "
                       "exercises where possible.", **wide(st.button))

    html(ui.phase_html(f"🔥 Warm-up · {workout.warmup_minutes} min — light cardio and dynamic "
                       "mobility for the muscles you will train."))
    cards = []
    for item in workout.items:
        chips = [ui.chip(item.muscle_group, "blue"), ui.chip(item.equipment),
                 ui.chip(item.difficulty, "green"), ui.chip(f"{item.match_percent:.0f}% match", "orange")]
        cards.append(ui.exercise_card(item.order, item.exercise_name, chips,
                                      f"{item.sets} × {item.reps}",
                                      f"Rest {item.rest_seconds} s · ~{item.est_minutes:.1f} min"))
    html("".join(cards))
    html(ui.phase_html(f"🧘 Cool-down · {workout.cooldown_minutes} min — easy walking and "
                       "static stretches."))
    spare = profile.duration - workout.total_minutes
    if spare >= max(5.0, 0.15 * profile.duration):
        st.info(f"About {spare:.0f} min of your {profile.duration}-minute slot is left unplanned. "
                f"Training volume is capped on purpose (at most "
                f"{MAX_SETS_BY_LEVEL.get(profile.fitness_level)} sets per exercise for "
                f"{profile.fitness_level} level and "
                f"{EXERCISE_COUNT_RANGE[profile.duration][1]} exercises for this duration) — "
                "use the spare time for a longer cool-down or extra mobility work.")

    st.markdown("#### Workout breakdown")
    frame = workout.to_frame()
    table = pd.DataFrame({
        "#": frame["order"], "Exercise": frame["exercise_name"], "Sets": frame["sets"],
        "Reps": frame["reps"], "Rest": frame["rest_seconds"].map(lambda s: f"{s} s"),
        "Est. Duration": frame["est_minutes"].map(lambda m: f"{m:.1f} min"),
        "Match": frame["match_percent"]})
    st.dataframe(table, hide_index=True, column_config={
        "Match": st.column_config.ProgressColumn(format="%.1f%%", min_value=0, max_value=100)},
        **wide(st.dataframe))
    st.caption(f"Estimated total: {workout.exercise_minutes:.1f} min of exercise + "
               f"{workout.warmup_minutes + workout.cooldown_minutes} min warm-up/cool-down = "
               f"**{workout.total_minutes:.0f} min** (requested {profile.duration} min). "
               "Time per exercise = sets × work time + rests between sets + ~1 min set-up.")
    with st.expander("📋 Exercise instructions & safety notes"):
        for item in workout.items:
            st.markdown(f"**{item.order}. {item.exercise_name}** — {item.instructions}  \n"
                        f"⚠️ _{item.safety_notes}_")
    st.download_button("⬇️ Download workout (CSV)", table.to_csv(index=False).encode("utf-8"),
                       file_name="personalized_workout.csv", mime="text/csv")
    html(ui.disclaimer_html("Sets, reps and rest periods are educational suggestions, not a "
                            "medical or professional training prescription. " + DISCLAIMER))


def render_why(profile: UserProfile, result: RecommendationResult, workout: Workout, top_n: int) -> None:
    top = top_frame(result, top_n)
    funnel = [
        ui.metric_card("Exercises in dataset", str(len(ENGINE.df)), "after cleaning", ui.MUTED),
        ui.metric_card("Passed hard filter", str(result.n_eligible), result.filter_stage),
        ui.metric_card("Top-N recommended", str(len(top)), "ranked by cosine", ui.ACCENT),
        ui.metric_card("In your workout", str(len(workout.items)), "diverse, time-fitted", ui.POSITIVE),
    ]
    html(ui.metric_grid(funnel))
    for message in result.warnings:
        st.info(message)
    st.markdown("Each match score is the cosine similarity between your profile vector and the "
                "exercise vector. Because each feature block is normalised and weighted, the "
                "score splits **exactly** into per-feature contributions:")
    plot(charts.contribution_stack(top), key="contrib_stack")

    names = top["exercise_name"].tolist()
    pick = st.selectbox("Inspect one recommendation", names, key="why_pick")
    row = top.loc[top["exercise_name"] == pick].iloc[0]
    left, right = st.columns([1, 1])
    with left:
        html(f'<div class="pw-score">#{int(row["rank"])} {escape(str(pick)).upper()} — '
             f'{row["match_percent"]:.1f}% MATCH</div>')
        st.markdown("**Why recommended:**")
        render_reasons(row, profile)
        breakdown = score_breakdown(row)
        breakdown.insert(1, "Weight", [ENGINE.encoder.weights[b] for b in BLOCK_ORDER])
        st.dataframe(breakdown, hide_index=True, column_config={
            "Block similarity": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)},
            **wide(st.dataframe))
        st.caption(f"Contributions add up to {breakdown['Contribution (%)'].sum():.1f}% "
                   f"(match score {row['match_percent']:.1f}%; small differences are rounding).")
    with right:
        st.markdown("**Similarity per feature block** (1 = identical)")
        plot(charts.block_similarity_radar(row), key="radar")


def render_statistics(profile: UserProfile, result: RecommendationResult, workout: Workout, top_n: int) -> None:
    frame = workout.to_frame()
    kcal = workout.total_calories(profile.weight_kg)
    total_sets = int(frame["sets"].sum())
    cards = [
        ui.metric_card("Total sets", str(total_sets), f"{len(frame)} exercises"),
        ui.metric_card("Average rest", f"{frame['rest_seconds'].mean():.0f} s", "between sets", ui.MUTED),
        ui.metric_card("Average match", f"{frame['match_percent'].mean():.1f}%", "workout exercises",
                       ui.ACCENT),
        ui.metric_card("Weekly time", f"{workout.total_minutes * profile.days_per_week:.0f} min",
                       f"{profile.days_per_week} sessions", ui.POSITIVE),
        ui.metric_card("Weekly calories", f"~{kcal * profile.days_per_week:,} kcal",
                       "approximate, educational", ui.ACCENT),
    ]
    html(ui.metric_grid(cards))
    c1, c2 = st.columns(2)
    with c1:
        plot(charts.time_breakdown_donut(workout), key="time_donut")
    with c2:
        plot(charts.per_exercise_bars(frame, "est_minutes", "Estimated minutes per exercise",
                                      "Minutes"), key="minutes_bar")
    c1, c2 = st.columns(2)
    with c1:
        plot(charts.per_exercise_bars(frame, "calories", "Approximate calories per exercise",
                                      "kcal", ui.ACCENT), key="kcal_bar")
    with c2:
        plot(charts.score_distribution(result.ranked, top_n), key="score_hist")
    plot(charts.weekly_projection(profile.days_per_week, workout.total_minutes, kcal), key="weekly")
    st.caption("Calories use MET-based approximations from the educational dataset scaled to "
               "your body weight; real energy expenditure varies widely between people.")


def render_rate_workout(profile: UserProfile, workout: Workout) -> None:
    st.markdown("#### How useful was this workout?")
    if st.session_state.feedback_done:
        st.success("Thanks! Your feedback was saved locally (data/feedback.json). Generate "
                   "another workout to rate it as well.")
        return
    with st.form("feedback_form", clear_on_submit=True):
        rating = st.select_slider("Usefulness rating", options=[1, 2, 3, 4, 5], value=4,
                                  format_func=lambda r: "★" * r + "☆" * (5 - r))
        liked = st.radio("Recommendations", ["👍 I liked these recommendations",
                                             "👎 I didn't like these recommendations"],
                         index=None, horizontal=True)
        comment = st.text_area("Optional feedback", placeholder="What would make this workout better?",
                               max_chars=500)
        submitted = st.form_submit_button("Submit feedback", type="primary")
    if submitted:
        record = {
            "rating": int(rating),
            "liked": None if liked is None else liked.startswith("👍"),
            "comment": comment.strip(),
            "history_id": st.session_state.history_id,
            "goal": profile.goal, "target": profile.target_muscle,
            "fitness_level": profile.fitness_level, "equipment": profile.equipment,
            "duration": profile.duration, "variation": workout.variation,
            "exercises": [i.exercise_name for i in workout.items],
        }
        try:
            feedback_store().append(record)
            st.session_state.feedback_done = True
            st.rerun()
        except OSError as exc:
            st.error(f"Feedback could not be saved ({exc}).")
    st.caption("Feedback is stored locally as groundwork for future personalisation. It is "
               "**not** used to retrain the model in this version.")


with tab_plan:
    render_form()
    profile, result, workout = (st.session_state.profile, st.session_state.result,
                                st.session_state.workout)
    if result is not None and workout is not None:
        top_n = int(st.session_state.top_n)
        html(ui.section_html("✓", "Your Workout Dashboard"))
        for message in result.warnings:
            st.warning(message)
        render_dashboard(profile, result, workout)
        t_rec, t_work, t_why, t_stats, t_rate = st.tabs(
            ["⭐ Recommended Exercises", "🗓️ Personalized Workout", "💡 Why These Exercises?",
             "📈 Workout Statistics", "💬 Rate This Workout"])
        with t_rec:
            render_recommendations(profile, result, top_n)
        with t_work:
            render_workout(profile, result, workout)
        with t_why:
            render_why(profile, result, workout, top_n)
        with t_stats:
            render_statistics(profile, result, workout, top_n)
        with t_rate:
            render_rate_workout(profile, workout)
    else:
        html(ui.banner_html("👋 Fill in the form (or load a sample profile from the sidebar) and "
                            "press <b>Generate My Workout</b>. Demo values: Age 21 · 175 cm · 70 kg · "
                            "Intermediate · Muscle Gain · Chest · Full Gym · 60 min · 5 days.", "info"))


# ============================================================================
# TAB 2 - Profile
# ============================================================================
with tab_profile:
    profile = st.session_state.profile
    html(ui.section_html("👤", "My Profile"))
    if profile is None:
        st.info("No profile yet. Generate a workout in **Plan My Workout** to see your profile here.")
    else:
        cards = [
            ui.metric_card("Age", f"{profile.age} yrs"),
            ui.metric_card("Height", f"{profile.height_cm:g} cm"),
            ui.metric_card("Weight", f"{profile.weight_kg:g} kg"),
            ui.metric_card("BMI", f"{profile.bmi:.1f}", profile.bmi_category, ui.MUTED),
            ui.metric_card("Fitness level", profile.fitness_level, accent=ui.POSITIVE),
            ui.metric_card("Goal", profile.goal, accent=ui.ACCENT),
            ui.metric_card("Target muscle", profile.target_muscle, accent=ui.ACCENT),
            ui.metric_card("Equipment", profile.equipment),
            ui.metric_card("Workout duration", f"{profile.duration} min"),
            ui.metric_card("Workout days", f"{profile.days_per_week} / week",
                           f"{profile.duration * profile.days_per_week} min weekly", ui.POSITIVE),
            ui.metric_card("Preferred type", profile.exercise_type),
        ]
        html(ui.metric_grid(cards))
        st.markdown("**BMI reference (WHO adult categories)** — informational only")
        bands = pd.DataFrame({"Category": ["Underweight", "Normal range", "Overweight", "Obese range"],
                              "BMI range": ["< 18.5", "18.5 – 24.9", "25.0 – 29.9", "≥ 30.0"]})
        bands["You"] = np.where(bands["Category"] == profile.bmi_category, "◀ you", "")
        st.dataframe(bands, hide_index=True, **wide(st.dataframe))
        st.caption("BMI does not distinguish muscle from fat and is not a medical assessment. "
                   "It is never used to choose exercises in this system.")


# ============================================================================
# TAB 3 - History
# ============================================================================
with tab_history:
    html(ui.section_html("🕘", "Workout History", "Saved locally in data/workout_history.json"))
    store = history_store()
    records = store.load()
    if store.last_warning:
        st.warning(f"⚠️ {store.last_warning}")
    if not records:
        st.info("No workouts saved yet. Every workout you generate is stored here automatically.")
    else:
        hist = pd.DataFrame([{
            "Date": r.get("date", ""), "Goal": r.get("goal", ""), "Target": r.get("target", ""),
            "Level": r.get("fitness_level", ""), "Duration (min)": r.get("duration", ""),
            "Est. time (min)": r.get("estimated_minutes", ""),
            "Exercises": len(r.get("exercises", []) or []),
            "Workout": ", ".join(e.get("name", "") for e in (r.get("exercises") or [])
                                 if isinstance(e, dict)),
        } for r in records]).iloc[::-1]
        goal_mode = hist["Goal"].mode()
        html(ui.metric_grid([
            ui.metric_card("Workouts generated", str(len(hist))),
            ui.metric_card("Most common goal", goal_mode.iloc[0] if len(goal_mode) else "—",
                           accent=ui.ACCENT),
            ui.metric_card("Planned minutes", f"{pd.to_numeric(hist['Est. time (min)'], errors='coerce').sum():.0f}",
                           "all saved workouts", ui.POSITIVE),
        ]))
        st.dataframe(hist, hide_index=True, **wide(st.dataframe))
        st.markdown("#### Recent workouts")
        for r in list(reversed(records))[:10]:
            title = (f"{r.get('date', '')} · {r.get('goal', '')} · {r.get('target', '')} · "
                     f"{r.get('duration', '')} min")
            with st.expander(title):
                for i, e in enumerate(r.get("exercises") or [], 1):
                    if isinstance(e, dict):
                        st.markdown(f"{i}. **{e.get('name', '')}** — {e.get('sets', '')} × "
                                    f"{e.get('reps', '')}, rest {e.get('rest_seconds', '')} s "
                                    f"({e.get('match_percent', 0):.0f}% match)")
        c1, c2 = st.columns([1, 1])
        c1.download_button("⬇️ Download history (JSON)",
                           json.dumps(records, indent=2, ensure_ascii=False).encode("utf-8"),
                           file_name="workout_history.json", mime="application/json")
        with c2:
            confirm = st.checkbox("I want to delete all saved history")
            if st.button("🗑️ Clear history", disabled=not confirm):
                store.clear()
                st.rerun()


# ============================================================================
# TAB 4 - Feedback summary
# ============================================================================
with tab_feedback:
    html(ui.section_html("💬", "Feedback", "Saved locally in data/feedback.json"))
    fstore = feedback_store()
    fb = fstore.load()
    if fstore.last_warning:
        st.warning(f"⚠️ {fstore.last_warning}")
    st.caption("Rate a workout from **Plan My Workout → Rate This Workout**. Feedback is "
               "collected as groundwork for future personalisation; it does not retrain the "
               "recommender in this version.")
    if not fb:
        st.info("No feedback yet.")
    else:
        fdf = pd.DataFrame(fb)
        ratings = pd.to_numeric(fdf.get("rating"), errors="coerce").dropna().astype(int)
        liked = fdf.get("liked", pd.Series(dtype=object)).dropna()
        html(ui.metric_grid([
            ui.metric_card("Responses", str(len(fdf))),
            ui.metric_card("Average rating", f"{ratings.mean():.2f} / 5" if len(ratings) else "—",
                           accent=ui.ACCENT),
            ui.metric_card("👍 share", f"{liked.astype(bool).mean():.0%}" if len(liked) else "—",
                           f"{len(liked)} thumbs given", ui.POSITIVE),
        ]))
        c1, c2 = st.columns([1, 2])
        with c1:
            if len(ratings):
                plot(charts.feedback_ratings(ratings), key="fb_ratings")
        with c2:
            show = fdf.reindex(columns=["timestamp", "rating", "liked", "goal", "target",
                                        "fitness_level", "comment"]).iloc[::-1]
            show["liked"] = show["liked"].map({True: "👍", False: "👎"}).fillna("—")
            st.dataframe(show, hide_index=True, **wide(st.dataframe))


# ============================================================================
# TAB 5 - Dataset explorer
# ============================================================================
with tab_data:
    df = ENGINE.df
    html(ui.section_html("📊", "Explore Exercise Dataset", "data/exercises.csv"))
    html(ui.metric_grid([
        ui.metric_card("Total exercises", str(len(df))),
        ui.metric_card("Muscle groups", str(df["muscle_group"].nunique()), accent=ui.ACCENT),
        ui.metric_card("Equipment types", str(df["equipment"].nunique()), accent=ui.POSITIVE),
        ui.metric_card("Exercise types", str(df["exercise_type"].nunique())),
        ui.metric_card("Avg duration", f"{df['duration_minutes'].mean():.1f} min", accent=ui.MUTED),
    ]))
    with st.expander("🧹 Data preprocessing report"):
        st.json(CLEANING_REPORT.as_dict())
        st.caption("Steps: required-column check → category normalisation (aliases, case) → "
                   "duplicate removal → invalid numeric values replaced by the median of the same "
                   "exercise type → missing text filled → derived features (goal list, timed flag, "
                   "kcal/min, keyword tags).")

    st.markdown("#### Filter")
    c1, c2, c3 = st.columns(3)
    f_muscle = c1.multiselect("Muscle group", sorted(df["muscle_group"].unique()), key="dx_muscle")
    f_equip = c2.multiselect("Equipment", sorted(df["equipment"].unique()), key="dx_equip")
    f_diff = c3.multiselect("Difficulty", FITNESS_LEVELS, key="dx_diff")
    c1, c2, c3 = st.columns(3)
    f_goal = c1.multiselect("Goal (any listed)", GOALS, key="dx_goal")
    f_type = c2.multiselect("Exercise type", sorted(df["exercise_type"].unique()), key="dx_type")
    f_text = c3.text_input("Search by name", key="dx_text", placeholder="e.g. press, squat, plank")

    mask = np.ones(len(df), dtype=bool)
    for values, col in ((f_muscle, "muscle_group"), (f_equip, "equipment"),
                        (f_diff, "difficulty"), (f_type, "exercise_type")):
        if values:
            mask &= df[col].isin(values).to_numpy()
    if f_goal:
        wanted = set(f_goal)
        mask &= df["goals_list"].map(lambda g: bool(wanted.intersection(g))).to_numpy(dtype=bool)
    if f_text.strip():
        mask &= df["exercise_name"].str.contains(f_text.strip(), case=False, regex=False).to_numpy(dtype=bool)
    view = df.loc[mask]
    st.markdown(f"**{len(view)} exercises** match the filters.")
    cols = ["exercise_id", "exercise_name", "muscle_group", "equipment", "difficulty", "goal",
            "exercise_type", "duration_minutes", "calories_burned", "sets", "reps", "rest_seconds"]
    st.dataframe(view[cols], hide_index=True, height=320, **wide(st.dataframe))
    st.download_button("⬇️ Download filtered data (CSV)", view[cols].to_csv(index=False).encode("utf-8"),
                       file_name="exercises_filtered.csv", mime="text/csv")
    st.caption("Calories are approximate educational values (MET-based estimates for a ~70 kg "
               "person), not medically accurate measurements.")

    st.markdown("#### Visual analytics")
    if view.empty:
        st.info("No exercises match the current filters — clear a filter to see charts.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            plot(charts.count_bar(view["muscle_group"], "Exercises by muscle group", horizontal=True),
                 key="dx_muscle_chart")
        with c2:
            plot(charts.difficulty_donut(view["difficulty"]), key="dx_diff_chart")
        c1, c2 = st.columns(2)
        with c1:
            plot(charts.goal_bar(view["goals_list"]), key="dx_goal_chart")
        with c2:
            plot(charts.count_bar(view["equipment"], "Equipment distribution", ui.ACCENT),
                 key="dx_equip_chart")
        c1, c2 = st.columns(2)
        with c1:
            plot(charts.duration_histogram(view), key="dx_dur_chart")
        with c2:
            plot(charts.calories_scatter(view), key="dx_kcal_chart")
        res = st.session_state.result
        if res is not None:
            plot(charts.score_distribution(res.ranked, int(st.session_state.top_n)),
                 key="dx_score_chart")
        else:
            st.caption("Generate a workout to also see the recommendation match-score distribution.")


# ============================================================================
# TAB 6 - Evaluation
# ============================================================================
@st.cache_data(show_spinner="Running demonstration evaluation…")
def _run_evaluation(_engine: WorkoutRecommender, n_profiles: int, dataset_key: int):
    from evaluation.evaluate import (cosine_verification, evaluate_grid,
                                     evaluate_sample_users)
    return (cosine_verification(_engine), evaluate_sample_users(_engine, k=5),
            evaluate_grid(_engine, ks=(5, 10), max_profiles=n_profiles))


with tab_eval:
    html(ui.section_html("🧪", "Evaluation"))
    html(ui.banner_html(
        "<b>Demonstration / Test Evaluation.</b> Genuine Precision@K and Recall@K need relevance "
        "labels from real users or trainers, which this project does not have. The numbers below "
        "use a clearly labelled <b>synthetic</b> relevance set built from transparent rules "
        "(evaluation/test_relevance_labels.csv). They show how the metrics are computed and that "
        "the pipeline behaves as designed — they are <b>not</b> real-world accuracy.", "demo"))
    c1, c2, c3 = st.columns(3)
    with c1:
        st.latex(r"\text{Precision@K}=\frac{|\text{Rec}_K \cap \text{Rel}|}{K}")
    with c2:
        st.latex(r"\text{Recall@K}=\frac{|\text{Rec}_K \cap \text{Rel}|}{|\text{Rel}|}")
    with c3:
        st.latex(r"\cos(A,B)=\frac{A\cdot B}{\lVert A\rVert\,\lVert B\rVert}")

    n_profiles = st.select_slider("Profiles sampled from the 4,860-profile grid",
                                  options=[100, 300, 600, 1200], value=300, key="eval_n")
    if st.button("▶️ Run demonstration evaluation", type="primary"):
        st.session_state.eval_ready = n_profiles
    if st.session_state.get("eval_ready"):
        try:
            cos_df, users_df, grid_df = _run_evaluation(
                ENGINE, int(st.session_state.eval_ready), len(ENGINE.df))
        except Exception as exc:  # pragma: no cover - defensive
            st.error(f"Evaluation failed: {exc}")
        else:
            st.markdown("**1 · Cosine similarity check** — manual formula vs scikit-learn "
                        "for the demo user's Top-5")
            st.dataframe(cos_df.round(4), hide_index=True, **wide(st.dataframe))
            st.markdown("**2 · Sample users (K = 5)**")
            st.dataframe(users_df[["Profile", "n_relevant", "ours_precision", "ours_recall",
                                   "ours_ndcg", "filter_precision", "random_precision"]].round(3),
                         hide_index=True, **wide(st.dataframe))
            st.markdown(f"**3 · Profile grid** ({int(st.session_state.eval_ready)} sampled profiles)")
            st.dataframe(grid_df.round(3), hide_index=True, **wide(st.dataframe))
            c1, c2 = st.columns(2)
            with c1:
                plot(charts.evaluation_bars(grid_df, "Precision@K"), key="ev_p")
            with c2:
                plot(charts.evaluation_bars(grid_df, "Recall@K"), key="ev_r")
            st.caption("Baselines: Random = no personalisation; Filter only = random picks from the "
                       "hard-filtered pool (no cosine ranking). Because the synthetic labels use the "
                       "same attributes the recommender sees, these numbers are optimistic.")
    else:
        saved = EVAL_RESULTS_DIR / "demo_grid_metrics.csv"
        if saved.exists():
            st.markdown("**Saved results** from `python evaluation/evaluate.py` (full 4,860-profile "
                        "grid, demonstration labels):")
            try:
                st.dataframe(pd.read_csv(saved).round(3), hide_index=True, **wide(st.dataframe))
            except (OSError, pd.errors.ParserError):
                st.warning("Saved results file could not be read; press the button to recompute.")
        st.caption("Press the button to recompute live in the browser.")


# ============================================================================
# TAB 7 - How it works
# ============================================================================
with tab_how:
    html(ui.section_html("ℹ️", "How It Works", "Content-based filtering + cosine similarity"))
    st.markdown("""
1. **User input → user profile.** The form is validated and turned into a profile.
2. **Feature extraction.** Every exercise is encoded once into 8 feature blocks — goal (multi-hot),
   target muscle (one-hot), equipment characteristics, fitness level, exercise type, duration
   (soft bins), difficulty and a **TF-IDF** keyword profile.
3. **User feature vector.** Your answers are mapped into the *same* feature space
   (e.g. target *Chest* also gives partial weight to *Triceps* and *Shoulders*).
4. **Content-based filtering.** A hard eligibility filter removes unsuitable exercises
   (equipment you don't have, difficulty above your level, …). If too few remain, the filter is
   relaxed step by step so the result is never empty.
5. **Cosine similarity** between your vector and each eligible exercise vector → match score.
6. **Ranking → Top-N** recommendations.
7. **Workout generator** picks a diverse, time-fitted subset (MMR re-ranking) and prescribes
   sets, reps and rest.
""")
    st.latex(r"\cos(U,E)=\frac{U\cdot E}{\lVert U\rVert\lVert E\rVert}"
             r"=\frac{\sum_b w_b\,\cos(U_b,E_b)}{\lVert U\rVert\,\lVert E\rVert}")
    st.caption("Each block b is L2-normalised and multiplied by √w_b before concatenation, so the "
               "cosine becomes a weighted average of per-block similarities — which is what makes "
               "every score explainable.")
    c1, c2 = st.columns([1, 1])
    with c1:
        plot(charts.feature_weight_bar(ENGINE.encoder.weights), key="how_weights")
    with c2:
        dims = pd.DataFrame({"Feature block": [BLOCK_LABELS[b] for b in BLOCK_ORDER],
                             "Weight": [ENGINE.encoder.weights[b] for b in BLOCK_ORDER],
                             "Dimensions": [len(ENGINE.encoder.vocab[b]) for b in BLOCK_ORDER]})
        st.dataframe(dims, hide_index=True, **wide(st.dataframe))
        st.caption(f"Total vector length: {ENGINE.X.shape[1]} dimensions. High weight: goal, "
                   "target muscle, equipment, fitness level. Medium: type, duration, difficulty.")
    st.markdown("#### Hard-filter relaxation cascade")
    st.dataframe(pd.DataFrame({
        "Stage": range(1, len(FILTER_STAGES) + 1), "Name": FILTER_STAGES,
        "What it allows": [
            "Equipment available · difficulty at/below level · target muscle (or conditioning work "
            "for Weight Loss/Endurance) · exercise type compatible with the goal",
            "Difficulty widened by one step (Beginner may see Intermediate; Advanced may also see Beginner)",
            "Also allows synergist muscles and any exercise type",
            "Any exercise usable with your equipment (equipment is never relaxed)"]}),
        hide_index=True, **wide(st.dataframe))

    profile = st.session_state.profile
    if profile is not None:
        st.markdown("#### Your profile as a feature vector")
        vec = ENGINE.encoder.encode_user(profile).ravel()
        names = ENGINE.encoder.feature_names()
        nz = pd.DataFrame({"Feature": names, "Value": vec})
        nz = nz.loc[nz["Value"].abs() > 1e-9]
        nz.insert(0, "Block", nz["Feature"].str.split(":").str[0].map(BLOCK_LABELS))
        nz["Feature"] = nz["Feature"].str.split(":", n=1).str[1]
        st.dataframe(nz.round(4), hide_index=True, height=300, **wide(st.dataframe))
        st.caption(f"{len(nz)} of {len(vec)} dimensions are non-zero for your profile "
                   f"(weighted values, ‖U‖ = {np.linalg.norm(vec):.3f}).")
    else:
        st.caption("Generate a workout to see your own profile vector here.")
