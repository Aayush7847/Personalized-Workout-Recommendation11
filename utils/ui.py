"""
Presentation helpers for the Streamlit app: theme CSS and small HTML
components (hero banner, metric cards, chips, workout cards).

Only presentation lives here - no recommendation logic. All user / dataset
text is HTML-escaped before it is inserted into markup.
"""
from __future__ import annotations

from html import escape

# ----------------------------------------------------------------------------
# Design tokens
# ----------------------------------------------------------------------------
INK = "#172033"
CANVAS = "#F3F5F9"
SURFACE = "#FFFFFF"
PRIMARY = "#2347D6"
ACCENT = "#FF6B2C"
POSITIVE = "#1F9D6B"
WARNING = "#C98A00"
MUTED = "#5B6478"
LINE = "#E1E5EE"

# Categorical palette for charts (ordered, colour-blind friendly-ish)
CHART_COLORS = ["#2347D6", "#FF6B2C", "#1F9D6B", "#7A4DD8", "#E0A100",
                "#0E9AA7", "#D6457A", "#5B6478", "#8DB600", "#A0522D"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800&family=Barlow:wght@400;500;600&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, p, li, label {{
  font-family: 'Barlow', 'Segoe UI', system-ui, -apple-system, sans-serif;
}}
h1, h2, h3, h4 {{
  font-family: 'Barlow Condensed', 'Arial Narrow', 'Segoe UI', sans-serif !important;
  letter-spacing: .01em; color: {INK};
}}
.block-container {{ padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1280px; }}

/* ---------- hero ---------- */
.pw-hero {{
  background: radial-gradient(1200px 300px at 85% -40%, rgba(255,107,44,.35), transparent 60%),
              linear-gradient(120deg, {INK} 0%, #1E2C55 60%, {PRIMARY} 140%);
  border-radius: 18px; padding: 28px 32px 24px; color: #fff; margin-bottom: 18px;
  box-shadow: 0 10px 30px rgba(23,32,51,.18);
}}
.pw-hero .eyebrow {{ font-size: .78rem; letter-spacing: .22em; text-transform: uppercase;
  color: #FFB48F; font-weight: 600; }}
.pw-hero h1 {{ color: #fff !important; font-size: 3.1rem; font-weight: 800; line-height: 1;
  margin: .2rem 0 .3rem; text-transform: uppercase; padding: 0; }}
.pw-hero .sub {{ font-size: 1.15rem; color: #D9E0F5; margin-bottom: 14px; }}
.pw-pipeline {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }}
.pw-pipeline span.step {{ background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.18);
  border-radius: 999px; padding: 3px 11px; font-size: .8rem; color: #EEF2FF; }}
.pw-pipeline span.arrow {{ color: #FF9A6B; font-size: .8rem; }}

/* ---------- section titles ---------- */
.pw-section {{ display: flex; align-items: baseline; gap: 10px; margin: 18px 0 8px; }}
.pw-section .num {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 800; color: #fff;
  background: {ACCENT}; border-radius: 8px; padding: 1px 9px; font-size: 1rem; }}
.pw-section .title {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 700;
  font-size: 1.45rem; text-transform: uppercase; color: {INK}; letter-spacing: .02em; }}
.pw-section .hint {{ color: {MUTED}; font-size: .9rem; }}

/* ---------- metric cards ---------- */
.pw-metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(118px, 1fr));
  gap: 12px; margin: 6px 0 14px; }}
.pw-card {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 14px;
  padding: 12px 13px 10px; border-top: 4px solid var(--accent, {PRIMARY});
  box-shadow: 0 1px 2px rgba(23,32,51,.04); }}
.pw-card .label {{ font-size: .72rem; letter-spacing: .12em; text-transform: uppercase;
  color: {MUTED}; font-weight: 600; }}
.pw-card .value {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 700;
  font-size: 1.45rem; color: {INK}; line-height: 1.15; margin-top: 2px;
  word-break: keep-all; overflow-wrap: normal; hyphens: none; }}
.pw-card .subv {{ font-size: .82rem; color: {MUTED}; margin-top: 2px; }}

/* ---------- chips ---------- */
.pw-chip {{ display: inline-block; border-radius: 999px; padding: 2px 10px; margin: 2px 4px 2px 0;
  font-size: .78rem; font-weight: 600; background: #EEF1FA; color: {INK}; border: 1px solid {LINE}; }}
.pw-chip.blue {{ background: #E6ECFF; color: {PRIMARY}; border-color: #CBD6FF; }}
.pw-chip.orange {{ background: #FFEEE5; color: #B5441A; border-color: #FFD3BD; }}
.pw-chip.green {{ background: #E3F5EE; color: #146C4A; border-color: #BFE6D5; }}

/* ---------- workout list ---------- */
.pw-ex {{ display: grid; grid-template-columns: 48px 1fr auto; gap: 14px; align-items: center;
  background: {SURFACE}; border: 1px solid {LINE}; border-radius: 14px; padding: 12px 16px;
  margin-bottom: 10px; }}
.pw-ex .n {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 800; font-size: 1.6rem;
  color: #fff; background: {INK}; border-radius: 12px; text-align: center; padding: 4px 0; }}
.pw-ex .name {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 700; font-size: 1.3rem;
  color: {INK}; text-transform: uppercase; line-height: 1.1; }}
.pw-ex .rx {{ text-align: right; }}
.pw-ex .rx .big {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 800; font-size: 1.7rem;
  color: {PRIMARY}; line-height: 1; }}
.pw-ex .rx .small {{ font-size: .8rem; color: {MUTED}; }}
.pw-phase {{ background: #FFF6F1; border: 1px dashed #FFC3A6; border-radius: 12px; padding: 9px 16px;
  margin-bottom: 10px; color: #8A3A15; font-size: .9rem; }}

/* ---------- reasons ---------- */
.pw-reason {{ padding: 3px 0; font-size: .95rem; }}
.pw-reason .ic {{ display: inline-block; width: 1.4em; font-weight: 700; }}
.pw-reason.match .ic {{ color: {POSITIVE}; }}
.pw-reason.partial .ic {{ color: {WARNING}; }}
.pw-reason.note .ic {{ color: {ACCENT}; }}
.pw-score {{ font-family: 'Barlow Condensed', sans-serif; font-weight: 800; font-size: 1.5rem;
  color: {PRIMARY}; }}

/* ---------- banners ---------- */
.pw-banner {{ border-radius: 12px; padding: 12px 16px; margin: 6px 0 12px; font-size: .92rem;
  border: 1px solid; }}
.pw-banner.demo {{ background: #FFF8E1; border-color: #F2D27A; color: #6B4E00; }}
.pw-banner.info {{ background: #EEF2FF; border-color: #CBD6FF; color: #1D2F86; }}
.pw-disclaimer {{ font-size: .8rem; color: {MUTED}; border-left: 3px solid {ACCENT};
  padding: 6px 10px; background: #fff; border-radius: 0 8px 8px 0; }}

/* ---------- streamlit tweaks ---------- */
div[data-testid="stForm"] {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 16px;
  padding: 18px 20px 6px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
div[data-testid="stExpander"] {{ background: {SURFACE}; border-radius: 12px; }}
[data-testid="stSidebar"] {{ background: {SURFACE}; }}
div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {{
  font-weight: 700; letter-spacing: .03em; }}
@media (max-width: 640px) {{
  .pw-hero h1 {{ font-size: 2.2rem; }}
  .pw-ex {{ grid-template-columns: 40px 1fr; }}
  .pw-ex .rx {{ grid-column: 2; text-align: left; }}
}}
</style>
"""

PIPELINE_STEPS = ["User Input", "User Profile", "Feature Extraction", "Content-Based Filtering",
                  "Feature Vectors", "Cosine Similarity", "Ranking", "Top-N",
                  "Personalized Workout"]


def hero_html() -> str:
    steps = '<span class="arrow">→</span>'.join(
        f'<span class="step">{escape(s)}</span>' for s in PIPELINE_STEPS)
    return (
        '<div class="pw-hero">'
        '<div class="eyebrow">Recommendation Systems · Mini-Project</div>'
        '<h1>Personalized Workout</h1>'
        '<div class="sub">AI-Assisted Recommendation System — content-based filtering '
        'with cosine similarity</div>'
        f'<div class="pw-pipeline">{steps}</div>'
        '</div>'
    )


def section_html(num: str, title: str, hint: str = "") -> str:
    hint_html = f'<span class="hint">{escape(hint)}</span>' if hint else ""
    return (f'<div class="pw-section"><span class="num">{escape(num)}</span>'
            f'<span class="title">{escape(title)}</span>{hint_html}</div>')


def metric_card(label: str, value: str, sub: str = "", accent: str = PRIMARY) -> str:
    sub_html = f'<div class="subv">{escape(sub)}</div>' if sub else ""
    return (f'<div class="pw-card" style="--accent:{accent}">'
            f'<div class="label">{escape(label)}</div>'
            f'<div class="value">{escape(value)}</div>{sub_html}</div>')


def metric_grid(cards: list[str]) -> str:
    return f'<div class="pw-metrics">{"".join(cards)}</div>'


def chip(text: str, tone: str = "") -> str:
    return f'<span class="pw-chip {tone}">{escape(str(text))}</span>'


def exercise_card(order: int, name: str, chips: list[str], big: str, small: str) -> str:
    return (f'<div class="pw-ex"><div class="n">{order}</div>'
            f'<div><div class="name">{escape(name)}</div><div>{"".join(chips)}</div></div>'
            f'<div class="rx"><div class="big">{escape(big)}</div>'
            f'<div class="small">{escape(small)}</div></div></div>')


def phase_html(text: str) -> str:
    return f'<div class="pw-phase">{escape(text)}</div>'


def reason_html(status: str, icon: str, text: str) -> str:
    return (f'<div class="pw-reason {status}"><span class="ic">{escape(icon)}</span>'
            f'{escape(text)}</div>')


def banner_html(text: str, kind: str = "info") -> str:
    return f'<div class="pw-banner {kind}">{text}</div>'


def disclaimer_html(text: str) -> str:
    return f'<div class="pw-disclaimer">⚠️ {escape(text)}</div>'
