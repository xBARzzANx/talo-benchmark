"""
Implements the Chapter 6 analysis pipeline: turns the scored Phase 3 results
into the tables and chart used in the thesis. Reads the latest
results/phase3_scores_*.jsonl (written by evaluate_phase3.py) plus the
Phase 1/2/3 raw-result caches (for cost/latency, which evaluate_phase3.py
does not carry on the scored rows), and writes everything to analysis/output/:

  - results_table.md / results_table.tex   5 conditions x 4 classes + overall
  - cost_latency.md                        full per-condition cost and mean
                                            generation latency (Section 6:
                                            "if this condition were run from
                                            scratch", not just this run's
                                            marginal/new-call cost)
  - variance_worst_case.md                 per class, spread and worst case
                                            across the five conditions
  - quality_per_cost.md                    overall_score / total_cost_usd
  - results_chart.svg                      grouped bar chart (matplotlib)

Cost basis: SQL/anomaly generation cost uses the *actual* logged token
counts from the raw cache (exact, not estimated), aggregated across all
four task classes. KPI/Report judge cost uses the same documented
approximation as run_phase3.py's dry-run estimate (src.utils.phase3.
JUDGE_EST_*), since llm_judge.py does not log judge-call tokens.
`condition_cost_latency()` lives in src/utils/phase3.py and is also used
by app.py's cost comparison chart, so both stay consistent by construction.

Usage:
  python analysis/phase3_analysis.py
  python analysis/phase3_analysis.py --scores-file results/phase3_scores_20260815_0148.jsonl
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.phase3 import (
    TASK_CLASSES, CONDITIONS, CONDITION_IDS, CLASS_LABELS, CONDITION_COLORS,
    load_pricing, condition_cost_latency, find_latest,
)

OUTPUT_DIR = REPO_ROOT / "analysis" / "output"


# --------------------------------------------------------------------------
# Load scored results
# --------------------------------------------------------------------------

def load_scores(scores_file: str | None) -> list[dict]:
    if scores_file:
        path = Path(scores_file)
    else:
        path = find_latest("phase3_scores_*.jsonl")
        if not path:
            raise SystemExit("No results/phase3_scores_*.jsonl found -- run evaluate_phase3.py first.")
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def condition_class_means(scored: list[dict]) -> dict[tuple, float]:
    """(condition, task_class) -> mean normalized_score."""
    by_cell = defaultdict(list)
    for s in scored:
        if s.get("normalized_score") is not None:
            by_cell[(s["condition"], s["task_class"])].append(s["normalized_score"])
    return {k: sum(v) / len(v) for k, v in by_cell.items() if v}


def condition_overall(class_means: dict[tuple, float]) -> dict[str, float | None]:
    """condition -> unweighted macro-mean of its four class means (Section 6, E1)."""
    overall = {}
    for cid in CONDITION_IDS:
        vals = [class_means.get((cid, tc)) for tc in TASK_CLASSES]
        overall[cid] = sum(vals) / len(vals) if all(v is not None for v in vals) else None
    return overall


# --------------------------------------------------------------------------
# Variance / worst-case per class
# --------------------------------------------------------------------------

def variance_worst_case(class_means: dict[tuple, float]) -> dict[str, dict]:
    """Per task class: spread and worst case across the five conditions."""
    out = {}
    for tc in TASK_CLASSES:
        values = {cid: class_means[(cid, tc)] for cid in CONDITION_IDS if (cid, tc) in class_means}
        if len(values) < 2:
            continue
        worst_cid = min(values, key=values.get)
        best_cid = max(values, key=values.get)
        out[tc] = {
            "mean": statistics.mean(values.values()),
            "stdev": statistics.pstdev(values.values()),
            "worst_condition": worst_cid, "worst_value": values[worst_cid],
            "best_condition": best_cid, "best_value": values[best_cid],
            "range": values[best_cid] - values[worst_cid],
        }
    return out


# --------------------------------------------------------------------------
# File output: results table (MD + LaTeX)
# --------------------------------------------------------------------------

def write_results_table(class_means: dict, overall: dict) -> tuple[str, str]:
    md_lines = ["| Condition | " + " | ".join(CLASS_LABELS[tc] for tc in TASK_CLASSES) + " | Overall |",
                "|---|" + "---|" * (len(TASK_CLASSES) + 1)]
    tex_lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{Phase 3 results: normalized score by condition and task class.}",
        r"\label{tab:phase3-results}",
        r"\begin{tabular}{l" + "c" * (len(TASK_CLASSES) + 1) + "}", r"\toprule",
        "Condition & " + " & ".join(CLASS_LABELS[tc] for tc in TASK_CLASSES) + r" & Overall \\",
        r"\midrule",
    ]
    for cid in CONDITION_IDS:
        label = CONDITIONS[cid]["label"]
        cells = [class_means.get((cid, tc)) for tc in TASK_CLASSES]
        cell_strs = [f"{v:.3f}" if v is not None else "--" for v in cells]
        ov = overall.get(cid)
        ov_str = f"{ov:.3f}" if ov is not None else "--"
        md_lines.append(f"| {cid} -- {label} | " + " | ".join(cell_strs) + f" | **{ov_str}** |")
        tex_lines.append(f"{cid} ({label}) & " + " & ".join(cell_strs) + f" & \\textbf{{{ov_str}}} \\\\")
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(md_lines) + "\n", "\n".join(tex_lines) + "\n"


def write_cost_latency_md(cost_latency: dict) -> str:
    lines = ["| Condition | Generation Cost | Judge Cost | Total Cost | Mean Gen. Latency (s) | Calls (gen/judge) |",
             "|---|---|---|---|---|---|"]
    for cid in CONDITION_IDS:
        c = cost_latency[cid]
        lat = f"{c['mean_latency_s']:.2f}" if c["mean_latency_s"] is not None else "--"
        lines.append(
            f"| {cid} -- {CONDITIONS[cid]['label']} | ${c['generation_cost']:.4f} | ${c['judge_cost']:.4f} | "
            f"**${c['total_cost']:.4f}** | {lat} | {c['n_generation_calls']}/{c['n_judge_calls']} |"
        )
    lines.append("")
    lines.append(
        "Cost = full cost to run this condition from scratch across all 60 tasks (exact token "
        "counts from the Phase 1/2/3 cache), not just this run's new/marginal calls. Judge cost is "
        "an approximation (llm_judge.py does not log judge-call tokens); see src/utils/phase3.py."
    )
    return "\n".join(lines) + "\n"


def write_variance_md(variance: dict) -> str:
    lines = ["| Task Class | Mean | Std.Dev (N=5) | Worst Condition | Worst Value | Best Condition | Best Value | Range |",
             "|---|---|---|---|---|---|---|---|"]
    for tc in TASK_CLASSES:
        v = variance.get(tc)
        if not v:
            continue
        lines.append(
            f"| {CLASS_LABELS[tc]} | {v['mean']:.3f} | {v['stdev']:.3f} | {v['worst_condition']} | "
            f"{v['worst_value']:.3f} | {v['best_condition']} | {v['best_value']:.3f} | {v['range']:.3f} |"
        )
    lines.append("")
    lines.append(
        "Spread and worst case per task class across the five conditions (population std.dev, N=5). "
        "Supports the cost-efficiency-and-consistency argument for TALO (Section 6): a routing/prompt "
        "strategy that keeps worst-case class performance high is preferable even without a mean-score uplift."
    )
    return "\n".join(lines) + "\n"


def write_quality_per_cost_md(overall: dict, cost_latency: dict) -> str:
    lines = ["| Condition | Overall Score | Total Cost | Quality per $ |", "|---|---|---|---|"]
    rows = []
    for cid in CONDITION_IDS:
        ov = overall.get(cid)
        cost = cost_latency[cid]["total_cost"]
        qpc = ov / cost if ov is not None and cost > 0 else None
        rows.append((cid, ov, cost, qpc))
    for cid, ov, cost, qpc in rows:
        ov_str = f"{ov:.3f}" if ov is not None else "--"
        qpc_str = f"{qpc:.1f}" if qpc is not None else "--"
        lines.append(f"| {cid} -- {CONDITIONS[cid]['label']} | {ov_str} | ${cost:.4f} | {qpc_str} |")
    return "\n".join(lines) + "\n"


def write_chart_svg(class_means: dict, path: Path) -> None:
    """Grouped bar chart: one group per task class, one bar per condition."""
    ink = "#0b0b0b"
    muted = "#898781"
    gridline = "#e1e0d9"
    baseline = "#c3c2b7"
    surface = "#fcfcfb"

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    n_conditions = len(CONDITION_IDS)
    bar_width = 0.15
    group_gap = 0.02
    x = range(len(TASK_CLASSES))

    for i, cid in enumerate(CONDITION_IDS):
        offset = (i - (n_conditions - 1) / 2) * (bar_width + group_gap)
        values = [class_means.get((cid, tc), 0) for tc in TASK_CLASSES]
        bars = ax.bar(
            [xi + offset for xi in x], values, width=bar_width,
            color=CONDITION_COLORS[cid], label=f"{cid}", edgecolor=surface, linewidth=1,
        )
        for b, v in zip(bars, values):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6.5, color=ink)

    ax.set_xticks(list(x))
    ax.set_xticklabels([CLASS_LABELS[tc] for tc in TASK_CLASSES], color=ink, fontsize=10)
    ax.set_ylabel("Normalized score [0, 1]", color=ink, fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.tick_params(colors=muted)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(baseline)
    ax.yaxis.grid(True, color=gridline, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(
        [f"{cid} -- {CONDITIONS[cid]['label']}" for cid in CONDITION_IDS],
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=8,
    )
    ax.set_title("Phase 3: normalized score by condition and task class", color=ink, fontsize=11, pad=12)

    fig.tight_layout()
    fig.savefig(path, format="svg", facecolor=surface)
    plt.close(fig)


# --------------------------------------------------------------------------
# Console reporting
# --------------------------------------------------------------------------

def print_results_table(class_means: dict, overall: dict) -> None:
    print(f"\n{'Condition':<8}" + "".join(f"{CLASS_LABELS[tc]:>10}" for tc in TASK_CLASSES) + f"{'Overall':>10}")
    print("-" * (8 + 10 * (len(TASK_CLASSES) + 1)))
    for cid in CONDITION_IDS:
        row = f"{cid:<8}"
        for tc in TASK_CLASSES:
            v = class_means.get((cid, tc))
            row += f"{v:>10.3f}" if v is not None else f"{'--':>10}"
        ov = overall.get(cid)
        row += f"{ov:>10.3f}" if ov is not None else f"{'--':>10}"
        print(row)


def print_cost_latency(cost_latency: dict) -> None:
    print(f"\n{'Condition':<8}{'Gen. Cost':>12}{'Judge Cost':>12}{'Total':>12}{'Mean Lat.(s)':>14}{'Calls':>12}")
    print("-" * 70)
    for cid in CONDITION_IDS:
        c = cost_latency[cid]
        gen_cost_str = f"${c['generation_cost']:.4f}"
        judge_cost_str = f"${c['judge_cost']:.4f}"
        total_cost_str = f"${c['total_cost']:.4f}"
        lat = f"{c['mean_latency_s']:.2f}" if c["mean_latency_s"] is not None else "--"
        calls_str = f"{c['n_generation_calls']}/{c['n_judge_calls']}"
        print(f"{cid:<8}{gen_cost_str:>12}{judge_cost_str:>12}{total_cost_str:>12}{lat:>14}{calls_str:>12}")


def print_variance(variance: dict) -> None:
    print(f"\n{'Class':<10}{'Mean':>8}{'StdDev':>10}{'Worst':>8}{'(cond)':>8}{'Best':>8}{'(cond)':>8}{'Range':>8}")
    print("-" * 68)
    for tc in TASK_CLASSES:
        v = variance.get(tc)
        if not v:
            continue
        print(f"{CLASS_LABELS[tc]:<10}{v['mean']:>8.3f}{v['stdev']:>10.3f}"
              f"{v['worst_value']:>8.3f}{v['worst_condition']:>8}"
              f"{v['best_value']:>8.3f}{v['best_condition']:>8}{v['range']:>8.3f}")


# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores-file", default=None, help="Defaults to the latest results/phase3_scores_*.jsonl")
    args = parser.parse_args()

    scored = load_scores(args.scores_file)
    class_means = condition_class_means(scored)
    overall = condition_overall(class_means)
    pricing = load_pricing()
    cost_latency = condition_cost_latency(pricing)
    variance = variance_worst_case(class_means)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_md, results_tex = write_results_table(class_means, overall)
    (OUTPUT_DIR / "results_table.md").write_text(results_md, encoding="utf-8")
    (OUTPUT_DIR / "results_table.tex").write_text(results_tex, encoding="utf-8")
    (OUTPUT_DIR / "cost_latency.md").write_text(write_cost_latency_md(cost_latency), encoding="utf-8")
    (OUTPUT_DIR / "variance_worst_case.md").write_text(write_variance_md(variance), encoding="utf-8")
    (OUTPUT_DIR / "quality_per_cost.md").write_text(write_quality_per_cost_md(overall, cost_latency), encoding="utf-8")
    write_chart_svg(class_means, OUTPUT_DIR / "results_chart.svg")

    print(f"Analyse-Outputs geschrieben nach: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f.name}")

    print("\n=== Haupttabelle: 5 Bedingungen x 4 Klassen + Gesamt ===")
    print_results_table(class_means, overall)

    print("\n=== Kosten / Latenz je Bedingung ===")
    print_cost_latency(cost_latency)

    print("\n=== Varianz und Worst-Case je Klasse ===")
    print_variance(variance)


if __name__ == "__main__":
    main()
