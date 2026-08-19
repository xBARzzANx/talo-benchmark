"""
Implements the TALO Streamlit demo described in Section 5.6: an end-to-end
walkthrough of the pipeline (task classification -> routing -> prompt
assembly -> optional live call) for a single natural-language analytics
request, plus a cost comparison against the Chapter 6 baseline conditions
(B1-B5). Serves as a figure in the thesis, so the interface is presentation-
grade: English throughout, no content hidden behind interaction that would
be invisible in a static screenshot (rationale and signal-score chart are
always expanded, never inside a collapsed expander).

Task classification and routing are always dry-run (no cost). A live model
call only happens when the user explicitly clicks "Run live call" -- the
GUI equivalent of the [y/N] confirmation used by the CLI scripts.

Run: streamlit run app.py
"""
import html

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from talo.classifier import MIN_TOTAL_SCORE
from talo.orchestrator import TALOOrchestrator
from src.utils.phase3 import CONDITION_IDS, load_pricing, condition_cost_latency

st.set_page_config(page_title="TALO Demo", page_icon="\U0001F9ED", layout="wide")

# --------------------------------------------------------------------------
# Design tokens -- shared between the injected CSS and the matplotlib charts
# so the UI chrome and the figures read as one system.
# --------------------------------------------------------------------------
ACCENT = "#4F46E5"          # deep indigo -- the one thing that should draw the eye
SURFACE = "#FFFFFF"         # card surface
PAGE_BG = "#FAFAFA"
TEXT = "#18181B"
TEXT_SECONDARY = "#52525B"
MUTED = "#8A8B93"
BORDER = "#E4E4E7"
GRIDLINE = "#ECEDF0"

# Neutral ramp for the four baseline conditions (B1-B4); TALO (B5) alone
# gets the accent colour. Four distinct grey steps stay distinguishable in
# a black-and-white print of the thesis, and keep the reader's eye on the
# framework rather than the baselines.
BASELINE_RAMP = {"B1": "#D4D4D8", "B2": "#A1A1AA", "B3": "#71717A", "B4": "#52525B"}

CLASS_DISPLAY_LABELS = {
    "sql_generation": "SQL Generation",
    "anomaly_detection": "Anomaly Detection",
    "kpi_interpretation": "KPI Interpretation",
    "report_generation": "Report Generation",
}
STRATEGY_DISPLAY_LABELS = {
    "zero_shot": "Zero-Shot",
    "few_shot": "Few-Shot",
    "chain_of_thought": "Chain-of-Thought",
    "structured_output": "Structured Output",
}

EXAMPLE_QUERIES = {
    "-- Select an example request --": "",
    "SQL Generation": "Show me the total revenue by region for Q3.",
    "Anomaly Detection": (
        "Here is last week's daily order volume: 1240, 1198, 1305, 1276, 4890, "
        "1241, 1189. Anything I should look at?"
    ),
    "KPI Interpretation": (
        "Our conversion rate is 2.1% this quarter, down from 2.6%. "
        "What does that mean for us?"
    ),
    "Report Generation": "Summarise the key figures for the board.",
    "Compound / ambiguous": (
        "Pull the top ten products by margin and write it up for the leadership meeting."
    ),
}


# --------------------------------------------------------------------------
# CSS -- typography, layout, and card/rationale styling over native widgets.
# No hand-built HTML components stand in for what Streamlit already does;
# this only restyles the existing widgets.
# --------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

        .stApp {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background-color: {PAGE_BG};
        }}

        .block-container {{
            max-width: 900px;
            margin: 0 auto;
            padding-top: 3rem;
            padding-bottom: 5rem;
        }}

        .stApp h1 {{
            font-size: 2.1rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            color: {TEXT};
            margin-bottom: 0.2rem;
        }}

        [data-testid="stCaptionContainer"] p, .stApp small {{
            color: {TEXT_SECONDARY} !important;
        }}

        .stApp code, .stApp pre, [data-testid="stCodeBlock"] * {{
            font-family: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace !important;
        }}

        /* Section rhythm: generous whitespace instead of dividing lines. */
        .talo-section {{
            margin-top: 3.25rem;
        }}
        .talo-step-label {{
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: {ACCENT};
            margin: 0 0 0.3rem 0;
        }}
        .talo-section-title {{
            font-size: 1.2rem;
            font-weight: 500;
            color: {TEXT};
            margin: 0 0 1rem 0;
        }}

        /* Classification / routing cards -- subtle border, generous padding,
           no shadow. Targets the containers created with
           st.container(border=True, key=...). */
        .st-key-classification-card, .st-key-routing-card {{
            border: 1px solid {BORDER} !important;
            border-radius: 14px !important;
            padding: 1.6rem 1.8rem !important;
            box-shadow: none !important;
            background: {SURFACE};
        }}

        .talo-fallback-hint {{
            font-size: 0.82rem;
            color: {MUTED};
            margin-top: 0.5rem;
        }}

        /* Rationale: a calm, offset block with a left accent line, not a
           bordered expander -- always visible so it survives a screenshot. */
        .talo-rationale {{
            border-left: 3px solid {ACCENT};
            padding: 0.1rem 0 0.1rem 1.1rem;
            margin-top: 1.1rem;
        }}
        .talo-rationale-label {{
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: {MUTED};
            margin: 0 0 0.35rem 0;
        }}
        .talo-rationale-text {{
            font-size: 0.92rem;
            line-height: 1.65;
            color: {TEXT_SECONDARY};
            margin: 0;
        }}

        .talo-chart-label {{
            font-size: 0.82rem;
            font-weight: 500;
            color: {TEXT_SECONDARY};
            margin: 1.4rem 0 0.5rem 0;
        }}

        section[data-testid="stSidebar"] h2 {{
            font-size: 1rem;
            font-weight: 600;
            color: {TEXT};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(step_number: int, title: str) -> None:
    st.markdown(
        f'<div class="talo-section">'
        f'<p class="talo-step-label">Step {step_number:02d}</p>'
        f'<p class="talo-section-title">{html.escape(title)}</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


def fallback_reason(scores: dict[str, float]) -> str:
    """Which of the two fallback triggers fired, in the classifier's own terms."""
    total = sum(scores.values())
    if total < MIN_TOTAL_SCORE:
        return "fallback: evidence floor not met"
    return "fallback: confidence below threshold"


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------

def signal_score_chart(scores: dict[str, float], predicted_class: str) -> plt.Figure:
    """Horizontal bars, one per task class, so labels stay readable -- the
    predicted class in the accent colour, the others in a single neutral tone."""
    class_order = list(scores.keys())
    labels = [CLASS_DISPLAY_LABELS.get(tc, tc) for tc in class_order]
    values = [scores[tc] for tc in class_order]
    colors = [ACCENT if tc == predicted_class else "#D4D4D8" for tc in class_order]

    fig, ax = plt.subplots(figsize=(6.2, 2.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    y_pos = list(range(len(class_order)))
    bars = ax.barh(y_pos, values, color=colors, height=0.55, edgecolor=SURFACE, linewidth=1)
    max_val = max(values + [0.1])
    for b, v in zip(bars, values):
        ax.text(v + max_val * 0.03, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                 va="center", ha="left", fontsize=11, color=TEXT)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11.5, color=TEXT)
    ax.invert_yaxis()
    ax.set_xlim(0, max_val * 1.25)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    return fig


def cost_comparison_chart(cost_latency: dict[str, dict]) -> plt.Figure:
    """
    Generation cost per condition, TALO (B5) vs. the four baselines --
    aggregated across all four task classes (60 tasks total), consistent
    with analysis/output/cost_latency.md. Generation cost only: judge/
    evaluation cost is excluded (Design-Entscheidung E6) since it is
    fixed per task and does not vary with the routing/prompting choice
    this chart is illustrating.
    """
    fig, ax = plt.subplots(figsize=(6.8, 3.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    bar_labels = ["TALO" if cid == "B5" else cid for cid in CONDITION_IDS]
    costs = [cost_latency[cid]["generation_cost"] for cid in CONDITION_IDS]
    colors = [ACCENT if cid == "B5" else BASELINE_RAMP[cid] for cid in CONDITION_IDS]

    bars = ax.bar(bar_labels, costs, color=colors, width=0.6, edgecolor=SURFACE, linewidth=1)
    max_cost = max(costs)
    for b, c in zip(bars, costs):
        ax.text(b.get_x() + b.get_width() / 2, c + max_cost * 0.02, f"${c:.4f}",
                 ha="center", va="bottom", fontsize=11.5, color=TEXT)

    ax.set_ylabel("Generation cost across all 60 tasks (USD)", color=TEXT_SECONDARY, fontsize=11)
    ax.set_ylim(0, max_cost * 1.18)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=12)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.9, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

@st.cache_resource
def get_orchestrator() -> TALOOrchestrator:
    return TALOOrchestrator()


@st.cache_resource
def get_cost_data() -> dict[str, dict]:
    """
    Per-condition generation cost, aggregated over all four task classes and
    all 60 tasks -- the same computation behind analysis/output/cost_latency.md
    (src.utils.phase3.condition_cost_latency), not a per-class estimate.
    """
    return condition_cost_latency(load_pricing())


# --------------------------------------------------------------------------

def main() -> None:
    inject_css()

    st.title("TALO -- Task-Aware LLM Optimizer")
    st.caption(
        "An interactive walkthrough of the TALO pipeline: task classification, "
        "routing, prompt assembly, and an optional live model call."
    )

    if "result" not in st.session_state:
        st.session_state.result = None
    if "live_output" not in st.session_state:
        st.session_state.live_output = None

    with st.sidebar:
        st.header("Input")
        example_choice = st.selectbox("Example request", list(EXAMPLE_QUERIES.keys()))
        query = st.text_area(
            "Analytics request", value=EXAMPLE_QUERIES[example_choice], height=120,
            placeholder='e.g. "Show revenue by region"',
        )
        classify_clicked = st.button("Classify & Route", type="primary", use_container_width=True)

    if classify_clicked and query.strip():
        st.session_state.result = get_orchestrator().run(query, dry_run=True)
        st.session_state.live_output = None

    result = st.session_state.result
    if result is None:
        st.info("Enter a request on the left and click **Classify & Route**.")
        return

    cls = result.classification

    section_header(1, "Task Classification")
    with st.container(border=True, key="classification-card"):
        col1, col2 = st.columns([1, 1])
        with col1:
            st.metric("Task class", CLASS_DISPLAY_LABELS.get(cls.task_class, cls.task_class))
            st.progress(min(max(cls.confidence, 0.0), 1.0), text=f"Confidence: {cls.confidence:.2f}")
            if cls.used_fallback:
                st.markdown(
                    f'<p class="talo-fallback-hint">{fallback_reason(cls.scores)}</p>',
                    unsafe_allow_html=True,
                )
        with col2:
            st.markdown('<p class="talo-chart-label">Signal scores per task class</p>', unsafe_allow_html=True)
            st.pyplot(signal_score_chart(cls.scores, cls.task_class))

    section_header(2, "Routing Decision")
    if result.routing:
        r = result.routing
        entry = get_orchestrator().router.get_alignment_matrix()[cls.task_class]
        with st.container(border=True, key="routing-card"):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Model", r.model_id)
            with col2:
                st.metric("Prompt strategy", STRATEGY_DISPLAY_LABELS.get(r.strategy, r.strategy))
            st.caption(
                f"Routing confidence: **{r.confidence}**  |  "
                f"Phase 1 score: {entry['phase1_score']} ({entry['metric']})  |  "
                f"Phase 2 score: {entry['phase2_score']}"
            )
            st.markdown(
                f'<div class="talo-rationale">'
                f'<p class="talo-rationale-label">Rationale (Alignment Matrix)</p>'
                f'<p class="talo-rationale-text">{html.escape(entry["rationale"])}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.error(f"Routing not possible: {result.error}")

    section_header(3, "Assembled Prompt")
    if result.prompt:
        st.code(result.prompt, language=None)
    else:
        st.warning(f"Prompt could not be assembled: {result.error}")

    section_header(4, "Model Output")
    live_col1, live_col2 = st.columns([1, 3])
    with live_col1:
        st.caption("Makes a real, billable API call -- nothing runs automatically.")
        if st.button("Run live call", disabled=result.prompt is None):
            with st.spinner("Calling model..."):
                st.session_state.live_output = get_orchestrator().run(query, dry_run=False)
    with live_col2:
        live = st.session_state.live_output
        if live is None:
            st.info("No live call yet -- dry run only, no cost incurred.")
        elif live.error:
            st.error(f"API error: {live.error}")
        else:
            st.code(live.output or "", language=None)

    section_header(5, "Cost Comparison: TALO vs. Baselines")
    st.caption(
        "Estimated generation cost across all 60 benchmark tasks (4 task classes), "
        "per condition from Chapter 6 (B1-B5) -- excluding evaluation/judge overhead, "
        "consistent with analysis/output/cost_latency.md, based on historical token "
        "counts and configs/pricing.yaml."
    )
    st.pyplot(cost_comparison_chart(get_cost_data()))


if __name__ == "__main__":
    main()
