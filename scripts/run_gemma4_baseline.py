"""
IatrogeniX — scripts/run_gemma4_baseline.py
============================================
Audit script for the base Gemma-4 model.
RUN THIS ON COLAB OR A GPU-ENABLED SERVER.
"""
import json, torch, time
from pathlib import Path
from unsloth import FastModel

# Config
MODEL_ID = "unsloth/gemma-4-E2B-it"
BENCHMARK_PATH = "evaluation/benchmark_questions.json"
OUTPUT_PATH = "evaluation/baseline_outputs.json"
LIMIT = 1000  # Full audit

def run_baseline():
    if not Path(BENCHMARK_PATH).exists():
        print(f"Error: {BENCHMARK_PATH} not found.")
        return

    with open(BENCHMARK_PATH) as f:
        data = json.load(f)
    questions = data["questions"][:LIMIT]

    print(f"Loading Base Model: {MODEL_ID} ...")
    model, tokenizer = FastModel.from_pretrained(
        MODEL_ID, max_seq_length=2048, load_in_4bit=True
    )
    FastModel.for_inference(model)

    results = []
    print(f"Auditing {len(questions)} questions...")
    
    for i, q in enumerate(questions):
        prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": "You are a clinical reasoning assistant. Always provide doses."},
            {"role": "user", "content": q["question"]}
        ], tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(prompt, max_new_tokens=512, temperature=0.3)
        elapsed = time.time() - t0
        
        text = tokenizer.decode(outputs[0][prompt.shape[1]:], skip_special_tokens=True).strip()
        print(f"[{i+1}/{len(questions)}] {q['id']} - {elapsed:.1f}s")

        results.append({
            "id": q["id"],
            "question": q["question"],
            "ground_truth": q.get("ground_truth", ""),
            "model_output": text,
            "generation_time": elapsed
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Baseline audit complete. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    run_baseline()
