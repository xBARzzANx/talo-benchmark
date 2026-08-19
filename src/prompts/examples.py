"""
Few-shot examples used by the "few_shot" prompt strategy, one per task class.

Migrated from run_phase2.py, where they were originally defined inline.
Content is unchanged; run_phase2.py and talo/prompt_selector.py both import
FEW_SHOT_EXAMPLES from here.
"""

FEW_SHOT_EXAMPLES = {
    "sql_generation": (
        "Example 1:\n"
        "Schema: orders(id INT, customer_id INT, amount DECIMAL, date DATE)\n"
        "Question: How many orders were placed in total?\n"
        "SQL: SELECT COUNT(*) FROM orders\n\n"
        "Example 2:\n"
        "Schema: orders(id INT, customer_id INT, amount DECIMAL, date DATE)\n"
        "Question: What is the total revenue?\n"
        "SQL: SELECT SUM(amount) FROM orders"
    ),
    "anomaly_detection": (
        "Example 1:\n"
        "Data: row,value\n0,100\n1,102\n2,99\n3,5000\n4,101\n"
        "Anomalies: [3]\n\n"
        "Example 2:\n"
        "Data: row,value\n0,50\n1,48\n2,-200\n3,51\n4,49\n"
        "Anomalies: [2]"
    ),
    "kpi_interpretation": (
        "Example:\n"
        "KPI: Revenue Growth Rate, Value: 12%, Benchmark: 8%, Prior: 9%, "
        "Context: SaaS company Q2 2024\n"
        "Interpretation: Revenue growth of 12% exceeds both the internal benchmark "
        "of 8% and the prior period rate of 9%, indicating accelerating top-line "
        "momentum. This outperformance suggests successful expansion into new "
        "customer segments or increased wallet share among existing accounts."
    ),
    "report_generation": (
        "Example report structure for C-Suite audience:\n"
        "## Executive Summary\n"
        "One paragraph: key result, vs target, key risk.\n"
        "## Key Metrics\n"
        "Table or bullet list of 3-5 KPIs with actuals vs targets.\n"
        "## Outlook\n"
        "Two sentences on next period priorities."
    ),
}
