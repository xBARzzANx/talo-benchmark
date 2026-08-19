"""LLM-as-Judge evaluator for KPI Interpretation and Report Generation."""
import json, os, re
import litellm

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "anthropic/claude-sonnet-4-5")

KPI_RUBRIC = """You are an expert evaluator for business analytics text.
Score the output on THREE criteria (1-5 each).
1. Factual correctness: Values, directions, and comparisons accurate?
2. Contextual appropriateness: Correct use of benchmark/prior period?
3. Communicative clarity: Clear for non-technical audience?
Respond ONLY with valid JSON:
{"factual_correctness": <1-5>, "contextual_appropriateness": <1-5>, "communicative_clarity": <1-5>, "reasoning": "<one sentence>"}"""

REPORT_RUBRIC = """You are an expert evaluator for business analytics reports.
Score the output on FOUR criteria (1-5 each).
1. Accuracy: All figures and trends correctly represented?
2. Completeness: All required sections and KPIs addressed?
3. Structure: Logical, professional structure?
4. Stakeholder appropriateness: Language suitable for stated audience?
Respond ONLY with valid JSON:
{"accuracy": <1-5>, "completeness": <1-5>, "structure": <1-5>, "stakeholder_appropriateness": <1-5>, "reasoning": "<one sentence>"}"""


def judge_output(task_class: str, task_input: str, model_output: str, ground_truth: str = "") -> dict:
    rubric = KPI_RUBRIC if task_class == "kpi_interpretation" else REPORT_RUBRIC
    gt_block = f"\nREFERENCE:\n{ground_truth}" if ground_truth else ""
    prompt = f"TASK INPUT:\n{task_input}\n\nMODEL OUTPUT:\n{model_output}{gt_block}\n\n{rubric}"
    try:
        response = litellm.completion(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
        )
        raw = response.choices[0].message.content
        raw = re.sub(r"```json|```", "", raw).strip()
        scores = json.loads(raw)
        criteria = [k for k in scores if k != "reasoning"]
        scores["mean_score"] = round(sum(scores[k] for k in criteria) / len(criteria), 2)
        scores["judge_model"] = JUDGE_MODEL
        return scores
    except Exception as e:
        return {"error": str(e), "mean_score": None, "judge_model": JUDGE_MODEL}