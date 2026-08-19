"""
Implements Appendix B of the thesis: renders the 12 prompt templates for
the three prompt strategies compared in Phase 2 (Few-Shot, Chain-of-Thought,
Structured Output) across the four task classes, as Markdown and LaTeX.

Zero-Shot is the Phase 1 baseline template, defined alongside these in
src/prompts/templates.py, but is not one of the three strategies Phase 2
evaluates against it -- so it is not reproduced in this appendix.

Usage:
  python export_appendix_b.py
"""
from pathlib import Path

from src.prompts.templates import TEMPLATES

OUTPUT_DIR = Path(__file__).resolve().parent / "analysis" / "output"
TASK_CLASSES = ["sql_generation", "anomaly_detection", "kpi_interpretation", "report_generation"]
STRATEGIES = ["few_shot", "chain_of_thought", "structured_output"]

TASK_CLASS_LABELS = {
    "sql_generation": "SQL Generation", "anomaly_detection": "Anomaly Detection",
    "kpi_interpretation": "KPI Interpretation", "report_generation": "Report Generation",
}
STRATEGY_LABELS = {
    "few_shot": "Few-Shot", "chain_of_thought": "Chain-of-Thought", "structured_output": "Structured Output",
}

INTRO = (
    "The 12 prompt templates for the three prompt strategies compared in Phase 2 "
    "(Few-Shot, Chain-of-Thought, Structured Output) across the four task classes. "
    "Zero-Shot is the Phase 1 baseline (see src/prompts/templates.py) and is not "
    "itself one of the compared strategies, so it is not reproduced here. Slot "
    "placeholders (e.g. {schema}, {question}) are filled at call time by "
    "talo/prompt_selector.py or the Phase 1/2/3 runners."
)


def write_markdown() -> str:
    lines = ["# Appendix B: Prompt Templates\n", INTRO + "\n"]
    for tc in TASK_CLASSES:
        lines.append(f"## {TASK_CLASS_LABELS[tc]}\n")
        for strategy in STRATEGIES:
            lines.append(f"### {STRATEGY_LABELS[strategy]}\n")
            lines.append("```")
            lines.append(TEMPLATES[tc][strategy])
            lines.append("```\n")
    return "\n".join(lines) + "\n"


def write_latex() -> str:
    lines = [r"\section{Appendix B: Prompt Templates}", "", INTRO, ""]
    for tc in TASK_CLASSES:
        lines.append(rf"\subsection{{{TASK_CLASS_LABELS[tc]}}}")
        for strategy in STRATEGIES:
            lines.append(rf"\subsubsection{{{STRATEGY_LABELS[strategy]}}}")
            lines.append(r"\begin{verbatim}")
            lines.append(TEMPLATES[tc][strategy])
            lines.append(r"\end{verbatim}")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUTPUT_DIR / "appendix_b_prompt_templates.md"
    tex_path = OUTPUT_DIR / "appendix_b_prompt_templates.tex"
    md_path.write_text(write_markdown(), encoding="utf-8")
    tex_path.write_text(write_latex(), encoding="utf-8")

    n = len(TASK_CLASSES) * len(STRATEGIES)
    print(f"{n} Prompt-Templates exportiert:")
    print(f"  {md_path}")
    print(f"  {tex_path}")


if __name__ == "__main__":
    main()
