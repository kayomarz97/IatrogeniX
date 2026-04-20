"""
IatrogeniX — evaluation/mass_eval.py
======================================
High-scale evaluation runner for 1,000 clinical questions.
Optimized for terminal status updates.

Usage:
  python3 evaluation/mass_eval.py --limit 1000
"""
from __future__ import annotations
import os, json, time, sys
from pathlib import Path
from tqdm import tqdm

# Add root to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from evaluation.eval import run_finetuned, build_comparison_report
from safety.validator import SafetyValidator

# ── Config ────────────────────────────────────────────────────────────────────
BENCHMARK_FILE = root_dir / "evaluation/benchmark_questions.json"
MASS_OUT       = root_dir / "evaluation/mass_eval_results.json"
LIVE_LOG       = root_dir / "evaluation/LIVE_LOG.txt"
FINAL_REPORT   = root_dir / "evaluation/mass_report.json"

def main():
    if not BENCHMARK_FILE.exists():
        print(f"Error: Benchmark file not found at {BENCHMARK_FILE}")
        return

    with open(BENCHMARK_FILE) as f:
        data = json.load(f)
    
    questions = data["questions"]
    limit = 1000
    qs = questions[:limit]
    
    print(f"\n🚀 STARTING MASS EVALUATION: {len(qs)} Clinical Cases")
    print(f"📊 Tracking progress in: {LIVE_LOG}\n")
    
    with open(LIVE_LOG, "w") as log:
        log.write(f"=== IatrogeniX Mass Eval Started at {time.ctime()} ===\n")
        log.write(f"Target: {len(qs)} questions\n\n")

    results = []
    # We use our own loop instead of run_finetuned to get better live logging
    from llama_cpp import Llama
    from inference.engine import build_gemma4_prompt, GEMMA4_TURN_START, GEMMA4_TURN_END
    
    model_path = root_dir / "models/iatrogenix-q5_k_m.gguf"
    llm = Llama(model_path=str(model_path), n_ctx=2048, n_gpu_layers=0, verbose=False)
    validator = SafetyValidator()

    pbar = tqdm(total=len(qs), desc="Clinical Audit", unit="case")
    
    for i, q in enumerate(qs):
        t0 = time.time()
        prompt = build_gemma4_prompt(q["question"])
        
        # Inference
        out = llm(prompt, max_tokens=512, temperature=0.3, stop=[GEMMA4_TURN_START, GEMMA4_TURN_END, "<eos>"])
        elapsed = time.time() - t0
        text = out["choices"][0]["text"].strip()
        
        # Safety Check
        v_result = validator.validate(text, question=q["question"])
        
        entry = {
            "id": q["id"],
            "question": q["question"],
            "ground_truth": q.get("ground_truth", ""),
            "model_output": text,
            "safety_status": v_result.status,
            "latency": round(elapsed, 2)
        }
        results.append(entry)
        
        # Live Logging
        with open(LIVE_LOG, "a") as log:
            status_char = "✅" if v_result.status == "safe" else "⚠️" if v_result.status == "warning" else "🚨"
            log.write(f"[{i+1}/{len(qs)}] {status_char} {q['id']} | {elapsed:.1f}s | Status: {v_result.status}\n")
            if v_result.status != "safe":
                for issue in v_result.issues:
                    log.write(f"   └─ {issue.severity.upper()}: {issue.description}\n")
        
        pbar.update(1)
        # Periodic Save
        if (i+1) % 50 == 0:
            with open(MASS_OUT, "w") as f:
                json.dump(results, f, indent=2)

    pbar.close()
    
    # Final Report
    print("\n✅ Evaluation Finished. Generating Final Comparison Report...")
    # Map back to eval.py format for the report builder
    published_scores = data.get("metadata", {}).get("published_model_scores", {})
    report = build_comparison_report([], results, published_scores)
    
    with open(FINAL_REPORT, "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"📊 Final Report: {FINAL_REPORT}")

if __name__ == "__main__":
    main()
