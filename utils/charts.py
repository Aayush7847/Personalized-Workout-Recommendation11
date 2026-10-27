"""
Plotly chart builders used by the Streamlit app.

Every chart is drawn from real data: the cleaned dataset, the scores the
recommender calculated, or the generated workout. Nothing is hard-coded.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from model.config import BLOCK_LABELS
from model.recommender import BLOCK_ORDER
from utils.ui import ACCENT, CHART_COLORS, INK, MUTED, POSITIVE, PRIMARY

FONT = "Barlow, Segoe UI, system-ui, sans-serif"


def _style(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=20, t=88 if title else 40, b=10),
        font=dict(family=FONT, color=INK, size=13),
        title=dict(text=title, x=0, y=0.99, yanchor="top", font=dict(size=17)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        # legend above the plot area (below the title) so it never hides axis titles
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom", font=dict(size=11)),
        hoverlabel=dict(font_family=FONT),
    )
    return fig


# ----------------------------------------------------------------------------
# Recommendation charts
# ----------------------------------------------------------------------------
def match_score_bar(top: pd.DataFrame) -> go.Figure:
    """Exercise vs match score (horizontal bars, rank 1 on top)."""
    data = top.iloc[::-1]
    colors = [ACCENT if r == 1 else PRIMARY for r in data["rank"]]
    fig = go.Figure(go.Bar(
        x=data["match_percent"], y=data["exercise_name"], orientation="h",
        marker_color=colors, text=[f"{v:.1f}%" for v in data["match_percent"]],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}<br>Match: %{x:.1f}%<extra></extra>"))
    fig.update_xaxes(range=[0, 105], title="Match score (cosine similarity × 100)")
    fig.update_yaxes(title=None)
    return _style(fig, height=max(260, 46 * len(top) + 80), title="Exercise vs Match Score")


def contribution_stack(top: pd.DataFrame) -> go.Figure:
    """Stacked bars: how much each feature block contributed to every score."""
    fig = go.Figure()
    names = top["exercise_name"].iloc[::-1]
    for i, b in enumerate(BLOCK_ORDER):
        fig.add_bar(name=BLOCK_LABELS[b], y=names, x=100 * top[f"contrib_{b}"].iloc[::-1],
                    orientation="h", marker_color=CHART_COLORS[i % len(CHART_COLORS)],
                    hovertemplate=f"{BLOCK_LABELS[b]}: %{{x:.1f}} pts<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Contribution to match score (points)", range=[0, 100])
    return _style(fig, height=max(300, 46 * len(top) + 120),
                  title="Score breakdown by feature (sums to the match score)")


def block_similarity_radar(row: pd.Series) -> go.Figure:
    """Per-feature cosine similarity for one exercise."""
    labels = [BLOCK_LABELS[b] for b in BLOCK_ORDER]
    values = [float(row[f"sim_{b}"]) for b in BLOCK_ORDER]
    fig = go.Figure(go.Scatterpolar(r=values + values[:1], theta=labels + labels[:1],
                                    fill="toself", line_color=PRIMARY,
                                    fillcolor="rgba(35,71,214,.18)"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], tickfont_size=10),
                                 domain=dict(x=[0.12, 0.88], y=[0.08, 0.92])), showlegend=False)
    fig = _style(fig, height=400)
    fig.update_layout(margin=dict(l=40, r=40, t=30, b=30))
    return fig


def score_distribution(ranked: pd.DataFrame, top_n: int) -> go.Figure:
    """Histogram of the match score of every eligible exercise, Top-N highlighted."""
    df = ranked.assign(Group=["Top-N" if r <= top_n else "Other eligible" for r in ranked["rank"]])
    fig = px.histogram(df, x="match_percent", color="Group", nbins=25,
                       color_discrete_map={"Top-N": ACCENT, "Other eligible": PRIMARY})
    fig.update_xaxes(title="Match score (%)")
    fig.update_yaxes(title="Exercises")
    return _style(fig, height=320, title="Match scores of all eligible exercises")


# ----------------------------------------------------------------------------
# Workout charts
# ----------------------------------------------------------------------------
def time_breakdown_donut(workout) -> go.Figure:
    labels = ["Warm-up"] + [i.exercise_name for i in workout.items] + ["Cool-down"]
    values = [workout.warmup_minutes] + [i.est_minutes for i in workout.items] + [workout.cooldown_minutes]
    colors = ["#C9D1E3"] + CHART_COLORS[: len(workout.items)] + ["#C9D1E3"]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=.58, marker_colors=colors,
                           textinfo="percent", sort=False,
                           hovertemplate="%{label}: %{value:.1f} min<extra></extra>"))
    fig.add_annotation(text=f"<b>{workout.total_minutes:.0f}</b><br>min", showarrow=False,
                       font=dict(size=22, family=FONT, color=INK))
    fig = _style(fig, height=360, title="Where the time goes")
    fig.update_layout(legend=dict(orientation="v", x=1.0, y=0.5, yanchor="middle", xanchor="left"),
                      margin=dict(t=50))
    return fig


def per_exercise_bars(frame: pd.DataFrame, column: str, title: str, axis: str,
                      color: str = PRIMARY) -> go.Figure:
    fig = go.Figure(go.Bar(x=frame["exercise_name"], y=frame[column], marker_color=color,
                           text=[f"{v:.0f}" for v in frame[column]], textposition="outside",
                           cliponaxis=False))
    fig.update_yaxes(title=axis)
    fig.update_xaxes(title=None, tickangle=-20)
    return _style(fig, height=330, title=title)


def weekly_projection(days: int, minutes: float, kcal: int) -> go.Figure:
    labels = [f"Day {d}" for d in range(1, days + 1)]
    fig = go.Figure()
    fig.add_bar(x=labels, y=[minutes] * days, name="Minutes", marker_color=PRIMARY)
    fig.add_scatter(x=labels, y=[kcal] * days, name="Approx. kcal", yaxis="y2",
                    mode="lines+markers", line=dict(color=ACCENT, width=3))
    fig.update_layout(yaxis=dict(title="Minutes"),
                      yaxis2=dict(title="Approx. kcal", overlaying="y", side="right",
                                  showgrid=False))
    return _style(fig, height=320, title=f"Weekly plan if repeated {days}× per week")


# ----------------------------------------------------------------------------
# Dataset analytics
# ----------------------------------------------------------------------------
def count_bar(series: pd.Series, title: str, color: str = PRIMARY, horizontal: bool = False) -> go.Figure:
    counts = series.value_counts()
    if horizontal:
        counts = counts.sort_values()
        fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker_color=color,
                               text=counts.values, textposition="outside", cliponaxis=False))
        fig.update_xaxes(title="Exercises")
    else:
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=color,
                               text=counts.values, textposition="outside", cliponaxis=False))
        fig.update_yaxes(title="Exercises")
    return _style(fig, height=360, title=title)


def difficulty_donut(series: pd.Series) -> go.Figure:
    counts = series.value_counts().reindex(["Beginner", "Intermediate", "Advanced"]).fillna(0)
    fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=.55,
                           marker_colors=[POSITIVE, PRIMARY, ACCENT], sort=False))
    return _style(fig, height=360, title="Exercises by difficulty")


def goal_bar(goals_lists: pd.Series) -> go.Figure:
    """Goal distribution: primary goal vs any goal (multi-label column)."""
    primary = goals_lists.map(lambda g: g[0]).value_counts()
    anyg = goals_lists.explode().value_counts()
    idx = anyg.index
    fig = go.Figure()
    fig.add_bar(x=idx, y=primary.reindex(idx).fillna(0), name="Primary goal", marker_color=PRIMARY)
    fig.add_bar(x=idx, y=anyg.reindex(idx), name="Listed goal (any)", marker_color="#AFC0F5")
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Exercises")
    return _style(fig, height=360, title="Exercises by goal")


def duration_histogram(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(df, x="duration_minutes", color="exercise_type", nbins=20,
                       color_discrete_sequence=CHART_COLORS)
    fig.update_xaxes(title="Exercise duration (minutes)")
    fig.update_yaxes(title="Exercises")
    return _style(fig, height=360, title="Exercise duration distribution")


def calories_scatter(df: pd.DataFrame) -> go.Figure:
    fig = px.scatter(df, x="duration_minutes", y="calories_burned", color="exercise_type",
                     hover_name="exercise_name", color_discrete_sequence=CHART_COLORS, opacity=.75)
    fig.update_xaxes(title="Duration (min)")
    fig.update_yaxes(title="Approx. calories (educational)")
    return _style(fig, height=360, title="Duration vs approximate calories")


def feature_weight_bar(weights: dict[str, float]) -> go.Figure:
    items = sorted(weights.items(), key=lambda kv: kv[1])
    total = sum(weights.values())
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[BLOCK_LABELS[k] for k, _ in items],
                           orientation="h", marker_color=[ACCENT if v >= 2.5 else PRIMARY for _, v in items],
                           text=[f"{v:g} ({100 * v / total:.0f}%)" for _, v in items],
                           textposition="outside", cliponaxis=False))
    fig.update_xaxes(title="Weight")
    return _style(fig, height=330, title="Feature weights")


def evaluation_bars(grid: pd.DataFrame, metric: str) -> go.Figure:
    fig = px.bar(grid, x="K", y=metric, color="System", barmode="group", text_auto=".2f",
                 color_discrete_sequence=[PRIMARY, "#AFC0F5", MUTED])
    fig.update_xaxes(type="category", title="K")
    fig.update_yaxes(range=[0, 1.05])
    return _style(fig, height=340, title=f"{metric} (demonstration labels)")


def feedback_ratings(ratings: pd.Series) -> go.Figure:
    counts = ratings.value_counts().reindex([1, 2, 3, 4, 5]).fillna(0)
    fig = go.Figure(go.Bar(x=[f"{i}★" for i in counts.index], y=counts.values, marker_color=ACCENT,
                           text=counts.values.astype(int), textposition="outside", cliponaxis=False))
    fig.update_yaxes(title="Responses")
    return _style(fig, height=300, title="Rating distribution")
