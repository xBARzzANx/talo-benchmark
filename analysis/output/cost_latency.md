| Condition | Generation Cost | Judge Cost | Total Cost | Mean Gen. Latency (s) | Calls (gen/judge) |
|---|---|---|---|---|---|
| B1 -- Static Single-Model | $0.0233 | $0.1035 | **$0.1268** | 3.06 | 60/30 |
| B2 -- Single-Model Top Performer | $0.0441 | $0.1035 | **$0.1476** | 5.91 | 60/30 |
| B3 -- Routing-Exclusive | $0.0349 | $0.1035 | **$0.1384** | 3.55 | 60/30 |
| B4 -- Prompt-Exclusive | $0.0205 | $0.1035 | **$0.1240** | 3.08 | 60/30 |
| B5 -- TALO | $0.0392 | $0.1035 | **$0.1427** | 3.88 | 60/30 |

Cost = full cost to run this condition from scratch across all 60 tasks (exact token counts from the Phase 1/2/3 cache), not just this run's new/marginal calls. Judge cost is an approximation (llm_judge.py does not log judge-call tokens); see src/utils/phase3.py.
