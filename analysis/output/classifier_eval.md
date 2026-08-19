# Classifier Evaluation

## Benchmark Tasks (N=60, ground truth = task_class)

Overall accuracy: **1.000** (60/60)

| Task Class | Accuracy | n |
|---|---|---|
| sql_generation | 1.000 | 15 |
| anomaly_detection | 1.000 | 15 |
| kpi_interpretation | 1.000 | 15 |
| report_generation | 1.000 | 15 |

### Confusion Matrix

| Ground Truth \ Predicted | sql_generation | anomaly_detection | kpi_interpretation | report_generation |
|---|---|---|---|---|
| sql_generation | 15 | 0 | 0 | 0 |
| anomaly_detection | 0 | 15 | 0 | 0 |
| kpi_interpretation | 0 | 0 | 15 | 0 |
| report_generation | 0 | 0 | 0 | 15 |

No misclassifications.


## Ambiguous Queries (N=8, ground truth = expected_class)

Overall accuracy: **1.000** (8/8)

| Query ID | Ambiguity Type | Ground Truth | Predicted | Confidence | Fallback | Correct |
|---|---|---|---|---|---|---|
| amb_001 | compound_request | sql_generation | sql_generation | 0.88 | False | yes |
| amb_002 | lexical_overlap | sql_generation | sql_generation | 0.81 | False | yes |
| amb_003 | granularity | report_generation | report_generation | 1.00 | False | yes |
| amb_004 | inline_data | kpi_interpretation | kpi_interpretation | 1.00 | False | yes |
| amb_005 | implicit_instruction | anomaly_detection | anomaly_detection | 1.00 | False | yes |
| amb_006 | multi_step | sql_generation | sql_generation | 0.59 | False | yes |
| amb_007 | underspecified | kpi_interpretation | kpi_interpretation | 0.00 | True | yes |
| amb_008 | compound_request | kpi_interpretation | kpi_interpretation | 1.00 | False | yes |
