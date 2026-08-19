# Appendix B: Prompt Templates

The 12 prompt templates for the three prompt strategies compared in Phase 2 (Few-Shot, Chain-of-Thought, Structured Output) across the four task classes. Zero-Shot is the Phase 1 baseline (see src/prompts/templates.py) and is not itself one of the compared strategies, so it is not reproduced here. Slot placeholders (e.g. {schema}, {question}) are filled at call time by talo/prompt_selector.py or the Phase 1/2/3 runners.

## SQL Generation

### Few-Shot

```
You are a SQL expert. Here are example question-to-SQL translations:

{examples}

Now translate:
Schema:
{schema}

Question: {question}

Return ONLY the SQL query.
```

### Chain-of-Thought

```
You are a SQL expert. Think step by step.

Schema:
{schema}

Question: {question}

Step 1: Identify relevant tables.
Step 2: Determine joins.
Step 3: Identify filters and aggregations.
Step 4: Write the final SQL.

Final SQL:
```

### Structured Output

```
You are a SQL expert. Analyze and return JSON.

Schema:
{schema}

Question: {question}

Return ONLY: {{"tables_used": [...], "joins": [...], "sql": "SELECT ..."}}
```

## Anomaly Detection

### Few-Shot

```
You are a data analyst. Examples:

{examples}

Now analyze:
{data}

Return ONLY a JSON list of anomalous row indices.
```

### Chain-of-Thought

```
You are a data analyst. Think step by step.

Dataset:
{data}

Step 1: Calculate mean/std per column.
Step 2: Flag rows >2.5 std deviations.
Step 3: Check for logical impossibilities.
Step 4: List anomalous indices.

Final answer (JSON list only):
```

### Structured Output

```
You are a data analyst.

Dataset:
{data}

Return ONLY: {{"anomalous_rows": [<indices>], "reasons": {{"<index>": "<reason>"}}}}
```

## KPI Interpretation

### Few-Shot

```
You are a business analyst. Examples:

{examples}

KPI: {kpi_name}
Value: {value}
Benchmark: {benchmark}
Prior period: {prior_period}
Context: {context}

Write a 2–4 sentence interpretation.
```

### Chain-of-Thought

```
You are a business analyst.

KPI: {kpi_name}
Value: {value}
Benchmark: {benchmark}
Prior period: {prior_period}
Context: {context}

Step 1: Above or below benchmark?
Step 2: Trend vs prior period?
Step 3: Business meaning?
Interpretation (no step labels, 2–4 sentences):
```

### Structured Output

```
You are a business analyst.

KPI: {kpi_name}
Value: {value}
Benchmark: {benchmark}
Prior period: {prior_period}
Context: {context}

Return ONLY: {{"status": "above/below/at target", "trend": "improving/declining/stable", "interpretation": "<2-4 sentences>"}}
```

## Report Generation

### Few-Shot

```
You are a business analyst. Example report:

{examples}

Now generate:
Data: {data}
Audience: {audience}
Sections: {sections}
```

### Chain-of-Thought

```
You are a business analyst.

Data: {data}
Audience: {audience}
Sections: {sections}

Step 1: Key message.
Step 2: Section structure.
Step 3: Write the report.

Report:
```

### Structured Output

```
You are a business analyst.

Data: {data}
Audience: {audience}
Sections: {sections}

Return ONLY valid JSON with one key per section name, each containing the section text.
```

