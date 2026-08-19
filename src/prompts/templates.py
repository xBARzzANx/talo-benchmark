"""Prompt strategy templates: 4 task classes × 4 strategies."""

TEMPLATES = {
    "sql_generation": {
        "zero_shot": (
            "You are a SQL expert. Given the database schema and a natural language question, "
            "write a valid SQL query that answers the question.\n\n"
            "Database Schema:\n{schema}\n\nQuestion: {question}\n\nReturn ONLY the SQL query."
        ),
        "few_shot": (
            "You are a SQL expert. Here are example question-to-SQL translations:\n\n{examples}\n\n"
            "Now translate:\nSchema:\n{schema}\n\nQuestion: {question}\n\nReturn ONLY the SQL query."
        ),
        "chain_of_thought": (
            "You are a SQL expert. Think step by step.\n\nSchema:\n{schema}\n\nQuestion: {question}\n\n"
            "Step 1: Identify relevant tables.\nStep 2: Determine joins.\n"
            "Step 3: Identify filters and aggregations.\nStep 4: Write the final SQL.\n\nFinal SQL:"
        ),
        "structured_output": (
            "You are a SQL expert. Analyze and return JSON.\n\nSchema:\n{schema}\n\nQuestion: {question}\n\n"
            'Return ONLY: {{"tables_used": [...], "joins": [...], "sql": "SELECT ..."}}'
        ),
    },
    "anomaly_detection": {
        "zero_shot": (
            "You are a data analyst. Identify anomalies in this dataset.\n\nDataset (CSV):\n{data}\n\n"
            "Return ONLY a JSON list of anomalous row indices (0-based). Example: [2, 7]"
        ),
        "few_shot": (
            "You are a data analyst. Examples:\n\n{examples}\n\n"
            "Now analyze:\n{data}\n\nReturn ONLY a JSON list of anomalous row indices."
        ),
        "chain_of_thought": (
            "You are a data analyst. Think step by step.\n\nDataset:\n{data}\n\n"
            "Step 1: Calculate mean/std per column.\nStep 2: Flag rows >2.5 std deviations.\n"
            "Step 3: Check for logical impossibilities.\nStep 4: List anomalous indices.\n\nFinal answer (JSON list only):"
        ),
        "structured_output": (
            "You are a data analyst.\n\nDataset:\n{data}\n\n"
            'Return ONLY: {{"anomalous_rows": [<indices>], "reasons": {{"<index>": "<reason>"}}}}'
        ),
    },
    "kpi_interpretation": {
        "zero_shot": (
            "You are a business analyst. Interpret this KPI for a business audience.\n\n"
            "KPI: {kpi_name}\nValue: {value}\nBenchmark: {benchmark}\n"
            "Prior period: {prior_period}\nContext: {context}\n\n"
            "Write a clear 2–4 sentence interpretation for non-technical stakeholders."
        ),
        "few_shot": (
            "You are a business analyst. Examples:\n\n{examples}\n\n"
            "KPI: {kpi_name}\nValue: {value}\nBenchmark: {benchmark}\n"
            "Prior period: {prior_period}\nContext: {context}\n\n"
            "Write a 2–4 sentence interpretation."
        ),
        "chain_of_thought": (
            "You are a business analyst.\n\n"
            "KPI: {kpi_name}\nValue: {value}\nBenchmark: {benchmark}\n"
            "Prior period: {prior_period}\nContext: {context}\n\n"
            "Step 1: Above or below benchmark?\nStep 2: Trend vs prior period?\n"
            "Step 3: Business meaning?\nInterpretation (no step labels, 2–4 sentences):"
        ),
        "structured_output": (
            "You are a business analyst.\n\n"
            "KPI: {kpi_name}\nValue: {value}\nBenchmark: {benchmark}\n"
            "Prior period: {prior_period}\nContext: {context}\n\n"
            'Return ONLY: {{"status": "above/below/at target", "trend": "improving/declining/stable", "interpretation": "<2-4 sentences>"}}'
        ),
    },
    "report_generation": {
        "zero_shot": (
            "You are a business analyst. Generate a concise analytics report.\n\n"
            "Data:\n{data}\nAudience: {audience}\nRequired sections: {sections}\n\n"
            "Write the report in professional business language."
        ),
        "few_shot": (
            "You are a business analyst. Example report:\n\n{examples}\n\n"
            "Now generate:\nData: {data}\nAudience: {audience}\nSections: {sections}"
        ),
        "chain_of_thought": (
            "You are a business analyst.\n\nData: {data}\nAudience: {audience}\nSections: {sections}\n\n"
            "Step 1: Key message.\nStep 2: Section structure.\nStep 3: Write the report.\n\nReport:"
        ),
        "structured_output": (
            "You are a business analyst.\n\nData: {data}\nAudience: {audience}\nSections: {sections}\n\n"
            "Return ONLY valid JSON with one key per section name, each containing the section text."
        ),
    },
}


def get_prompt(task_class: str, strategy: str, **kwargs) -> str:
    template = TEMPLATES[task_class][strategy]
    return template.format(**kwargs)
