"""
IatrogeniX — evaluation/eval.py
=================================
Phase 4: Failure analysis + comparison report.

Usage (on Colab after training):
  python evaluation/eval.py --mode baseline   # run baseline model
  python evaluation/eval.py --mode finetuned  # run fine-tuned GGUF
  python evaluation/eval.py --mode compare    # generate comparison_report.json
"""
from __future__ import annotations

import json, time, re, argparse, sys
from pathlib import Path
from typing import Optional
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Paths ─────────────────────────────────────────────────────────────────────
EVAL_DIR       = Path(__file__).parent
BENCHMARK_FILE = EVAL_DIR / "benchmark_questions.json"
BASELINE_OUT   = EVAL_DIR / "baseline_outputs.json"
FINETUNED_OUT  = EVAL_DIR / "finetuned_outputs.json"
REPORT_OUT     = EVAL_DIR / "comparison_report.json"

MODEL_BASE     = "unsloth/gemma-4-E2B-it"
GGUF_PATH      = "models/iatrogenix-q4_k_m.gguf"

SYSTEM_PROMPT  = (
    "You are a clinical reasoning assistant. Provide evidence-based answers "
    "with step-by-step reasoning. Always mention drug names with doses where applicable."
)

# ── Dose extraction ───────────────────────────────────────────────────────────
DOSE_RE = re.compile(
    r"(?:(?P<d1>[A-Za-z][a-z]+(?:\s[a-z]+){0,2})\s+"
    r"(?P<v1>\d+(?:\.\d+)?(?:[-–]\d+(?:\.\d+)?)?)\s*"
    r"(?P<u1>mg|mcg|g|mEq|units?|IU|mL)"
    r"|(?P<v2>\d+(?:\.\d+)?)\s*(?P<u2>mg|mcg|g|mEq|units?|IU|mL)"
    r"\s+of\s+(?P<d2>[A-Za-z][a-z]+(?:\s[a-z]+){0,2}))",
    re.I,
)
ABSOLUTE_RE = [re.compile(p, re.I) for p in [
    r"\balways\b", r"\bnever\b", r"\b100%\b",
    r"\bguaranteed\b", r"\bdefinitely\b", r"\binvariably\b",
]]
HEDGE_RE = [re.compile(p, re.I) for p in [
    r"\btypically\b", r"\boften\b", r"\busually\b",
    r"\bgenerally\b", r"\bmay\b", r"\bmight\b", r"\bconsidered\b",
]]

# ── Failure detectors ─────────────────────────────────────────────────────────
def detect_hallucination(model_output: str, ground_truth: str, known_drugs: set[str]) -> list[str]:
    """Return list of drugs in output not in GT and not in known drugs."""
    def drugs_in(text):
        tl = text.lower()
        return {n for n in known_drugs if n and re.search(rf"\b{re.escape(n)}\b", tl)}

    model_drugs = drugs_in(model_output)
    gt_drugs    = drugs_in(ground_truth)
    unknown     = model_drugs - gt_drugs - known_drugs
    # also flag fictional names
    fictional = {
        "medicinol","curicin","healitol","therapeutin","curemax",
        "fixitol","recoverin","remedizine","doctorex","healthagen",
    }
    found_fictional = {f for f in fictional if re.search(rf"\b{re.escape(f)}\b", model_output.lower())}
    return list(unknown | found_fictional)


def extract_doses(text: str) -> list[dict]:
    out = []
    for m in DOSE_RE.finditer(text):
        drug = (m.group("d1") or m.group("d2") or "").strip().lower()
        val_s = m.group("v1") or m.group("v2") or "0"
        unit  = (m.group("u1") or m.group("u2") or "").lower()
        val   = float(re.split(r"[-–]", val_s)[0])
        out.append({"drug": drug, "val": val, "val_s": val_s, "unit": unit})
    return out


def detect_dose_errors(model_output: str, ground_truth: str) -> list[str]:
    """Return list of dose discrepancy descriptions (>20% deviation)."""
    model_doses = {(d["drug"], d["unit"]): d for d in extract_doses(model_output)}
    gt_doses    = {(d["drug"], d["unit"]): d for d in extract_doses(ground_truth)}
    errors = []
    for key, md in model_doses.items():
        if key in gt_doses:
            gd = gt_doses[key]
            if gd["val"] > 0:
                pct = abs(md["val"] - gd["val"]) / gd["val"]
                if pct > 0.20:
                    errors.append(
                        f"{md['drug']}: model={md['val_s']}{md['unit']} "
                        f"GT={gd['val_s']}{gd['unit']} ({pct*100:.0f}% diff)"
                    )
    return errors


def detect_overconfidence(model_output: str) -> list[str]:
    """Return list of absolute statements without hedging."""
    flagged = []
    for pat in ABSOLUTE_RE:
        m = pat.search(model_output)
        if m:
            ctx = model_output[max(0, m.start()-60): m.end()+60]
            if not any(h.search(ctx) for h in HEDGE_RE):
                flagged.append(m.group(0))
    return flagged


# ── ROUGE-L simple scorer ─────────────────────────────────────────────────────
def lcs_length(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = [[0]*(n+1) for _ in range(2)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i%2][j] = dp[(i-1)%2][j-1]+1 if a[i-1]==b[j-1] else max(dp[(i-1)%2][j], dp[i%2][j-1])
    return dp[m%2][n]

def rouge_l(hyp: str, ref: str) -> float:
    h, r = hyp.lower().split(), ref.lower().split()
    if not h or not r:
        return 0.0
    lcs = lcs_length(h, r)
    p = lcs / len(h)
    rec = lcs / len(r)
    if p + rec == 0:
        return 0.0
    return 2 * p * rec / (p + rec)


# ── Baseline inference (Unsloth, Colab) ───────────────────────────────────────
def run_baseline(questions: list[dict], limit: Optional[int] = None) -> list[dict]:
    """Run baseline zero-shot inference on Gemma 4 E2B-it (no fine-tuning)."""
    from unsloth import FastModel
    import torch

    print(f"Loading baseline model: {MODEL_BASE} ...")
    model, tokenizer = FastModel.from_pretrained(
        MODEL_BASE, max_seq_length=2048, load_in_4bit=True, dtype=None
    )
    FastModel.for_inference(model)

    results = []
    qs = questions[:limit] if limit else questions
    for i, q in enumerate(qs):
        print(f"  [{i+1}/{len(qs)}] {q['id']} ...", end=" ", flush=True)
        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": q["question"]},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda" if torch.cuda.is_available() else "cpu")

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                inputs, max_new_tokens=512, temperature=0.3, top_p=0.9,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0
        out_ids = outputs[0][inputs.shape[1]:]
        text    = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        print(f"{elapsed:.1f}s")

        results.append({
            "id": q["id"],
            "question": q["question"],
            "ground_truth": q.get("ground_truth", q.get("correct_answer_text", "")),
            "model_output": text,
            "generation_time_seconds": round(elapsed, 2),
            "source_dataset": q.get("source_dataset", ""),
            "benchmark_category": q.get("benchmark_category", ""),
        })

    return results


# ── Fine-tuned inference (GGUF, llama.cpp) ───────────────────────────────────
def run_finetuned(questions: list[dict], gguf_path: str = GGUF_PATH,
                  limit: Optional[int] = None) -> list[dict]:
    from llama_cpp import Llama
    from inference.engine import build_gemma4_prompt, GEMMA4_TURN_START, GEMMA4_TURN_END

    p = Path(gguf_path)
    if not p.exists():
        raise FileNotFoundError(f"GGUF not found: {gguf_path}. Run train.py first.")

    print(f"Loading GGUF: {gguf_path} ...")
    llm = Llama(model_path=str(p), n_ctx=2048, n_gpu_layers=0, verbose=False)

    results = []
    qs = questions[:limit] if limit else questions
    for i, q in enumerate(qs):
        print(f"  [{i+1}/{len(qs)}] {q['id']} ...", end=" ", flush=True)
        prompt = build_gemma4_prompt(q["question"])
        t0 = time.time()
        out = llm(prompt, max_tokens=512, temperature=0.3, top_p=0.9,
                  stop=[GEMMA4_TURN_START, GEMMA4_TURN_END, "<eos>"], echo=False)
        elapsed = time.time() - t0
        text    = out["choices"][0]["text"].strip()
        print(f"{elapsed:.1f}s")

        results.append({
            "id": q["id"],
            "question": q["question"],
            "ground_truth": q.get("ground_truth", q.get("correct_answer_text", "")),
            "model_output": text,
            "generation_time_seconds": round(elapsed, 2),
            "source_dataset": q.get("source_dataset", ""),
            "benchmark_category": q.get("benchmark_category", ""),
        })

    return results


# ── Comparison report ─────────────────────────────────────────────────────────
def build_comparison_report(
    baseline: list[dict],
    finetuned: list[dict],
    published_scores: dict,
) -> dict:
    """Build per-question and summary comparison report."""
    from safety.validator import SafetyValidator
    validator = SafetyValidator()
    known_drugs = validator.drug_db.all_names()

    def analyse(outputs: list[dict]) -> tuple[list[dict], dict]:
        per_q  = []
        counts = Counter()
        for row in outputs:
            gt  = row.get("ground_truth", "")
            out = row.get("model_output", "")

            hall = detect_hallucination(out, gt, known_drugs)
            dose = detect_dose_errors(out, gt)
            over = detect_overconfidence(out)
            rl   = rouge_l(out, gt)
            ok   = not hall and not dose and not over

            issues = (
                [f"hallucination: {h}" for h in hall]
                + [f"dose_error: {d}" for d in dose]
                + [f"overconfidence: {o}" for o in over]
            )
            per_q.append({
                "id": row["id"],
                "issues": issues,
                "hallucinations": hall,
                "dose_errors": dose,
                "overconfident": over,
                "rouge_l": round(rl, 4),
                "correct": ok,
            })
            counts["hallucinations"] += len(hall)
            counts["dose_errors"]    += len(dose)
            counts["overconfident"]  += len(over)
            counts["correct"]        += int(ok)

        return per_q, dict(counts)

    # If baseline is missing, create a dummy baseline stats dict
    if not baseline:
        n = len(finetuned)
        base_sum = {
            "total": n,
            "hallucinations": 0,
            "dose_errors": 0,
            "overconfident": 0,
            "correct": 0,
            "is_fixed_reference": True
        }
        base_per = []
    else:
        base_per, base_sum = analyse(baseline)

    fine_per, fine_sum = analyse(finetuned)

    # Map by id for per-question improved flag
    fine_map = {r["id"]: r for r in fine_per}
    per_q_combined = []
    for bq in base_per:
        fq = fine_map.get(bq["id"], {})
        per_q_combined.append({
            "id": bq["id"],
            "baseline_issues": bq["issues"],
            "finetuned_issues": fq.get("issues", []),
            "baseline_rouge_l": bq["rouge_l"],
            "finetuned_rouge_l": fq.get("rouge_l", 0.0),
            "improved": not fq.get("issues") and bq["issues"] != [],
        })

    n = len(baseline)
    base_sum["total"] = n
    fine_sum["total"] = len(finetuned)

    return {
        "summary": {
            "baseline": base_sum,
            "finetuned": fine_sum,
            "improvement": {
                k: base_sum.get(k, 0) - fine_sum.get(k, 0)
                for k in ("hallucinations", "dose_errors", "overconfident")
            },
        },
        "published_model_scores_reference": published_scores,
        "per_question": per_q_combined,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="IatrogeniX Evaluation")
    parser.add_argument("--mode", choices=["baseline", "finetuned", "compare", "all"],
                        default="all")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit questions (for fast testing)")
    parser.add_argument("--gguf", default=GGUF_PATH)
    args = parser.parse_args()

    if not BENCHMARK_FILE.exists():
        print(f"Benchmark file not found: {BENCHMARK_FILE}")
        print("Run: python evaluation/benchmark_loader.py")
        sys.exit(1)

    with open(BENCHMARK_FILE) as f:
        data = json.load(f)
    questions        = data["questions"]
    published_scores = data.get("metadata", {}).get("published_model_scores", {})
    print(f"Loaded {len(questions)} benchmark questions")

    if args.mode in ("baseline", "all"):
        print("\n=== Phase 2: Baseline (zero-shot) ===")
        baseline = run_baseline(questions, args.limit)
        BASELINE_OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(BASELINE_OUT, "w") as f:
            json.dump(baseline, f, indent=2)
        print(f"Saved → {BASELINE_OUT}")

    if args.mode in ("finetuned", "all"):
        print("\n=== Phase 4: Fine-tuned model ===")
        finetuned = run_finetuned(questions, args.gguf, args.limit)
        with open(FINETUNED_OUT, "w") as f:
            json.dump(finetuned, f, indent=2)
        print(f"Saved → {FINETUNED_OUT}")

    if args.mode in ("compare", "all"):
        print("\n=== Comparison Report ===")
        if not FINETUNED_OUT.exists():
            print(f"Error: {FINETUNED_OUT} not found. Run --mode finetuned first.")
            sys.exit(1)

        ft = json.loads(FINETUNED_OUT.read_text())
        bl = []
        if BASELINE_OUT.exists():
            bl = json.loads(BASELINE_OUT.read_text())
        else:
            print("  ⚠️ INFO: Local baseline_outputs.json not found.")
            print("  Comparing directly against published research benchmarks.")

        report = build_comparison_report(bl, ft, published_scores)
        with open(REPORT_OUT, "w") as f:
            json.dump(report, f, indent=2)
        # Print summary
        s = report["summary"]
        print("\n" + "="*55)
        print("COMPARISON SUMMARY")
        print("="*55)
        print(f"{'Metric':25s} {'Baseline':>10} {'Fine-tuned':>12} {'Change':>8}")
        print("-"*55)
        for k in ("hallucinations", "dose_errors", "overconfident", "correct"):
            bv = s["baseline"].get(k, 0)
            fv = s["finetuned"].get(k, 0)
            ch = fv - bv
            sym = "↓" if (ch < 0 and k != "correct") or (ch > 0 and k == "correct") else "↑" if ch != 0 else "="
            if s["baseline"].get("is_fixed_reference"):
                 print(f"  {k:23s} {'N/A':>10} {fv:>12} {'-':>6} {sym}")
            else:
                 print(f"  {k:23s} {bv:>10} {fv:>12} {ch:>+6} {sym}")

        # Add Research Comparison Table
        print("\n" + "="*55)
        print("PUBLISHED RESEARCH BENCHMARKS (Gemma-4 / GPT-4 Ref)")
        print("="*55)
        print(f"{'Model':25s} {'MedQA':>10} {'MedMCQA':>10} {'MMLU':>8}")
        print("-"*55)
        for model, scores in report["published_model_scores_reference"].items():
            print(f"  {model[:23]:23s} {scores.get('medqa_usmle',0)*100:>9.1f}% {scores.get('medmcqa',0)*100:>9.1f}% {scores.get('mmlu_medical',0)*100:>7.1f}%")
        print(f"\nSaved → {REPORT_OUT}")


if __name__ == "__main__":
    main()
