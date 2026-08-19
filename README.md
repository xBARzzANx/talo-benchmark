# TALO Benchmark

**Task-Aware LLM Optimizer — Benchmark Dataset and Evaluation Pipeline**

This is the companion repository to the master's thesis
*"Task-Aware LLM Optimization: An Integrated Framework for Model Routing
and Prompt Strategy Selection in Data Analytics Contexts"*
(Hochschule Esslingen, M.Sc. Angewandte Informatik, 2026).

It contains the benchmark dataset, the evaluation pipeline, and the TALO
framework implementation, and is intended to let the results reported in
the thesis be independently reproduced and inspected.

---

## Overview

TALO evaluates five LLMs across four data analytics task classes using a
three-phase experimental design:

- **Phase 1**: All models under zero-shot conditions (baseline performance)
- **Phase 2**: Three prompt strategies (Few-Shot, Chain-of-Thought, Structured Output) applied to the top model per task class, evaluated against the Phase 1 Zero-Shot baseline
- **Phase 3**: The TALO framework itself, evaluated against four baseline conditions (Chapter 6)

**Task classes**: SQL Generation · Anomaly Detection · KPI Interpretation · Report Generation
**Models**: GPT-4.1-mini · Claude Haiku 4.5 · Gemini 2.5 Flash · Llama 3.1 8B · Mistral 7B Instruct
**Gold Standard**: Claude Sonnet 4.5 (non-competing reference)

---

## Repository Structure

```
talo-benchmark/
├── benchmark/                  # Benchmark dataset (public)
│   ├── sql_generation/
│   ├── anomaly_detection/
│   ├── kpi_interpretation/
│   ├── report_generation/
│   └── ambiguous_queries.jsonl # Classifier evaluation set (Chapter 6)
├── talo/                       # TALO framework: classifier, router, prompt selector, orchestrator
├── src/
│   ├── models/                 # Unified model client (LiteLLM)
│   ├── evaluators/             # Task-specific evaluators + extractor variants
│   ├── prompts/                # Prompt strategy templates + few-shot examples
│   └── utils/                  # I/O, Phase 3 condition/cache helpers
├── results/                    # Experiment outputs (gitignored for raw data)
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   └── phase3_exploratory/     # Ad-hoc runs outside the formal B1-B5 design
├── analysis/                   # Analysis scripts + notebooks
│   └── output/                 # Generated tables/charts (Markdown, LaTeX, SVG)
├── configs/                    # Model, pricing, and experiment configuration
├── scripts/                    # Maintenance/diagnostic scripts (not part of the reproduction chain)
├── tests/                      # Unit tests (pytest)
├── run_phase1.py                # Phase 1 runner
├── run_phase2.py                # Phase 2 runner
├── run_phase3.py                # Phase 3 runner (five conditions, cache-first)
├── run_talo.py                  # TALO CLI
├── app.py                       # TALO Streamlit demo
├── evaluate_phase1.py / evaluate_phase2.py / evaluate_phase3.py
├── verify_pricing.py            # Reproduces the reported Phase 1 cost figures
├── export_appendix_b.py         # Exports the prompt templates (Appendix B)
├── pytest.ini
├── LICENSE                      # MIT (code)
├── LICENSE-DATA                 # CC BY 4.0 (benchmark dataset)
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/xBARzzANx/talo-benchmark.git
cd talo-benchmark
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Install Ollama and pull local models
# https://ollama.ai
ollama pull llama3.1:8b
ollama pull mistral:7b-instruct

# Configure API keys
cp .env.example .env
# Edit .env with your API keys
```

The `.env` file is optional — API keys can instead be set directly as
environment variables (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, ...);
`src/models/model_client.py` reads them via `python-dotenv`, which falls
back to the process environment if no `.env` file is present.

---

## Running the Benchmark

```bash
# Verify all models are reachable
python scripts/check_connectivity.py

# Phase 1: All models, zero-shot, all task classes
python run_phase1.py

# Phase 1 for a single task class (useful for testing)
python run_phase1.py --task-class sql_generation

# Dry run (no API calls)
python run_phase1.py --dry-run
```

`pytest` (or `python -m pytest`) runs the unit test suite in `tests/` only —
it will not trigger `scripts/check_connectivity.py`'s live model calls.
`pytest.ini` sets `testpaths = tests` to scope collection, and
`pythonpath = .` so `src`/`talo` imports resolve under the bare `pytest`
entry point the same way they already do under `python -m pytest` (which
adds the current directory to `sys.path` on its own).

---

## Running TALO

TALO's classifier, router, prompt selector, and orchestrator live in
`talo/`. Two ways to use the assembled pipeline:

```bash
# CLI: classify, route, and preview the prompt for a single query (dry run by default)
python run_talo.py "Show me revenue by region"

# Add --live to make a real API call with the routed model/strategy
python run_talo.py "Show me revenue by region" --live

# Interactive demo: input field, classification confidence, routing rationale
# (from the Alignment Matrix), prompt preview, live output, cost comparison
streamlit run app.py
```

---

## Phase 3 — Framework Evaluation

Chapter 6 evaluates TALO against four baseline conditions, all cache-first
against the Phase 1/2 results wherever possible:

| ID | Condition | Model | Strategy |
|---|---|---|---|
| B1 | Static Single-Model | GPT-4.1-mini | Zero-Shot (all classes) |
| B2 | Single-Model Top Performer | Gemini 2.5 Flash | Zero-Shot (all classes) |
| B3 | Routing-Exclusive | optimal per class | Zero-Shot (all classes) |
| B4 | Prompt-Exclusive | GPT-4.1-mini | optimal per class |
| B5 | TALO | optimal per class | optimal per class |

```bash
# Preview cache coverage, new calls needed, and an itemized cost estimate -- no API calls
python run_phase3.py --dry-run

# Run a single condition (shows cost, asks for confirmation before any real call)
python run_phase3.py --condition B4

# Score all conditions (reuses cached LLM-as-Judge scores; --dry-run previews new judge calls)
python evaluate_phase3.py --run-id <run-id-from-run_phase3>

# Generate the results table (Markdown + LaTeX), cost/latency and variance analysis,
# and a grouped bar chart into analysis/output/
python analysis/phase3_analysis.py

# Classifier accuracy over the 60 benchmark tasks + the ambiguous query set
python analysis/classifier_eval.py
```

`verify_pricing.py` independently reproduces the Phase 1 cost figures
reported in the thesis from the token counts already stored in
`results/phase1/`, cross-checked against `configs/pricing.yaml`.

---

## Results

Chapter 6 main results table (normalized score, [0,1]):

| Condition | SQL | Anomaly | KPI | Report | **Overall** |
|---|---|---|---|---|---|
| B1 — Static Single-Model | 0.533 | 0.978 | 0.833 | 1.000 | **0.836** |
| B2 — Single-Model Top Performer | 0.667 | 0.960 | 0.895 | 0.954 | **0.869** |
| B3 — Routing-Exclusive | 0.667 | 0.978 | 0.950 | 1.000 | **0.899** |
| B4 — Prompt-Exclusive | 0.733 | 0.978 | 0.867 | 1.000 | **0.894** |
| B5 — TALO | 0.600 | 0.978 | 0.939 | 1.000 | **0.879** |

TALO (B5) does not outperform the single-dimension baselines B3 and B4 on
any of the three evaluated dimensions — quality, generation cost, and
worst-case consistency. The cause is the Alignment Matrix's sequential
derivation procedure (model chosen first from Phase 1 Zero-Shot, prompt
strategy chosen second only for that already-fixed model), which is not
guaranteed to find the joint optimum over the full model × strategy search
space.

---

## Benchmark Dataset

The benchmark dataset (`benchmark/`) is publicly available under
[CC BY 4.0](LICENSE-DATA), with one exception noted below.

### SQL Generation
SQL tasks are derived from the [BIRD benchmark](https://bird-bench.github.io/)
(Dev Set, original license applies — not covered by this repository's
CC BY 4.0 license, see `LICENSE-DATA`).
The SQLite database files are **not included** in this repository.

To run SQL generation tasks locally:
1. Download the BIRD Dev Set from https://bird-bench.github.io/
2. Copy the following files into `benchmark/sql_generation/dbs/`:
   - `california_schools.sqlite`
   - `financial.sqlite`
   - `thrombosis_prediction.sqlite`

### Other Task Classes
Anomaly Detection, KPI Interpretation, and Report Generation tasks are
original contributions of this thesis and require no additional downloads.

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@mastersthesis{shemari2026talo,
  title  = {Task-Aware LLM Optimization: An Integrated Framework for Model Routing 
             and Prompt Strategy Selection in Data Analytics Contexts},
  author = {Shemari, Barsan},
  school = {Hochschule Esslingen},
  year   = {2026}
}
```

---

## License

- Code: [MIT License](LICENSE)
- Benchmark data: [CC BY 4.0](LICENSE-DATA), with the BIRD-derived SQL
  generation tasks excluded (see "Benchmark Dataset" above)
