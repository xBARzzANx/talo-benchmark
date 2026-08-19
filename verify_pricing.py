"""
Verify that configs/pricing.yaml reproduces the Phase 1 cost figures reported
in Section 4.5.3 of the thesis.

Recomputes per-model costs from the token counts stored in the Phase 1 result
files and compares them against the reported values. If a model deviates, the
script reports the implied price ratio so the configuration can be adjusted.

The figures already published in Chapter 4 are the reference. Adjust
pricing.yaml to match them, not the other way around.

COST: 0 API calls.

Run from the repository root:  python verify_pricing.py
"""

import json
from collections import defaultdict
from pathlib import Path

import yaml

PHASE1_RUN_IDS = ["20260522_2030", "20260522_2219_gemini_retry"]
PRICING_PATH = Path("configs/pricing.yaml")


def load_pricing() -> dict:
    with open(PRICING_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_phase1_results() -> list[dict]:
    """Load and deduplicate Phase 1 results, matching evaluate_phase1.py."""
    results = []
    for run_id in PHASE1_RUN_IDS:
        path = Path(f"results/phase1/{run_id}/results.jsonl")
        if not path.exists():
            print(f"  WARNING: {path} not found")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))

    seen = {}
    for r in results:
        seen[(r["task_class"], r["task_id"], r["model_id"])] = r
    return [r for r in seen.values() if not r.get("error")]


def main() -> None:
    pricing = load_pricing()
    models = pricing["models"]
    reported = pricing["reported_phase1_costs"]
    tolerance = reported.get("tolerance", 0.01)

    results = load_phase1_results()
    print(f"Loaded {len(results)} Phase 1 results\n")

    tokens = defaultdict(lambda: {"input": 0, "output": 0, "calls": 0})
    for r in results:
        m = r["model_id"]
        tokens[m]["input"] += r.get("input_tokens") or 0
        tokens[m]["output"] += r.get("output_tokens") or 0
        tokens[m]["calls"] += 1

    print(f"{'Model':<22}{'Calls':>7}{'In tok':>10}{'Out tok':>10}"
          f"{'Computed':>11}{'Reported':>11}{'Delta':>9}")
    print("-" * 80)

    total_computed = 0.0
    deviations = []

    for model_id in sorted(tokens):
        t = tokens[model_id]
        cfg = models.get(model_id)
        if not cfg:
            print(f"{model_id:<22}  no pricing entry -- skipped")
            continue

        cost = (t["input"] / 1_000_000 * cfg["input"]
                + t["output"] / 1_000_000 * cfg["output"])
        total_computed += cost

        rep = reported["by_model"].get(model_id)
        rep_str = f"{rep:.2f}" if rep is not None else "--"
        delta_str = f"{cost - rep:+.3f}" if rep is not None else "--"

        print(f"{cfg['label']:<22}{t['calls']:>7}{t['input']:>10,}"
              f"{t['output']:>10,}{cost:>11.3f}{rep_str:>11}{delta_str:>9}")

        if rep is not None and abs(cost - rep) > tolerance:
            deviations.append((model_id, cfg, t, cost, rep))

    print("-" * 80)
    print(f"{'TOTAL':<22}{'':<7}{'':<10}{'':<10}"
          f"{total_computed:>11.3f}{reported['total']:>11.2f}"
          f"{total_computed - reported['total']:>+9.3f}")

    print()
    if not deviations and abs(total_computed - reported["total"]) <= tolerance:
        print("OK -- pricing.yaml reproduces the figures reported in Section 4.5.3.")
        print("No changes needed. Chapter 4 and Chapter 6 will be consistent.")
        return

    print("DEVIATION -- the configured prices do not reproduce Section 4.5.3.")
    print("The published Chapter 4 figures are the reference. Adjust the prices")
    print("below in configs/pricing.yaml so that the computed values match.\n")

    for model_id, cfg, t, cost, rep in deviations:
        ratio = rep / cost if cost > 0 else 0
        print(f"  {cfg['label']}")
        print(f"    computed {cost:.3f} vs reported {rep:.2f}"
              f"  (factor {ratio:.2f})")
        print(f"    current: input {cfg['input']}, output {cfg['output']}")
        print(f"    scaled:  input {cfg['input'] * ratio:.3f}, "
              f"output {cfg['output'] * ratio:.3f}")
        print(f"    NOTE: scaling both prices uniformly is a heuristic. Check the")
        print(f"          provider's published rates for the correct split first.")
        print()


if __name__ == "__main__":
    main()
