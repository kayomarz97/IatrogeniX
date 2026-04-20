"""
IatrogeniX — evaluation/hallucination_heatmap.py
================================================
Generates a 'Safety Heatmap' from mass evaluation results.
Shows failure rates per clinical subspecialty.
"""
import json
from pathlib import Path
from collections import defaultdict

def generate_heatmap(results_path: str, output_path: str):
    res_path = Path(results_path)
    if not res_path.exists():
        print(f"Error: {results_path} not found.")
        return

    with open(res_path) as f:
        results = json.load(f)

    # Group by category
    stats = defaultdict(lambda: {"total": 0, "blocked": 0, "warning": 0, "safe": 0})
    
    for r in results:
        # Extract category from ID (e.g., mmlu_clinical_knowledge_57)
        cat = "_".join(r["id"].split("_")[:-1])
        stats[cat]["total"] += 1
        stats[cat][r["safety_status"]] += 1

    # Generate Markdown Table
    md = "# IatrogeniX: Hallucination & Safety Heatmap\n\n"
    md += "| Subspecialty | Total Cases | Safe % | Warnings | Blocked (Critical) |\n"
    md += "| :--- | :---: | :---: | :---: | :---: |\n"
    
    for cat, s in sorted(stats.items(), key=lambda x: x[1]["total"], reverse=True):
        safe_pct = (s["safe"] / s["total"]) * 100
        md += f"| {cat} | {s['total']} | {safe_pct:.1f}% | {s['warning']} | **{s['blocked']}** |\n"
    
    md += "\n---\n*Generated from mass evaluation audit results.*"
    
    Path(output_path).write_text(md)
    print(f"Heatmap saved to {output_path}")

if __name__ == "__main__":
    generate_heatmap("evaluation/mass_eval_results.json", "evaluation/SAFETY_HEATMAP.md")
