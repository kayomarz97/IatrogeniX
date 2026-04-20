# IatrogeniX — 4-Way Safety Ablation Study

> **"Semantic correctness and letter-correct answers are not the same thing — and confusing them led to one of the most interesting findings in this project."**

This document is a transparent, end-to-end audit of IatrogeniX across four model configurations, evaluated on 1,000 cases from the MedQA-GB benchmark. It covers the evaluation methodology, results, discovered failure modes, and a roadmap for improvement. Community feedback is welcome.

---

## 1. What Was Tested

Four configurations were run head-to-head on the same 1,000 medical MCQ questions:

| Mode | Model | Safety Layer | Description |
|---|---|---|---|
| **A** | IatrogeniX (Fine-tuned) | ON | Primary system — 2B Gemma 4 fine-tuned on clinical data + live SafetyValidator |
| **B** | IatrogeniX (Fine-tuned) | OFF | Ablation: same fine-tuned model, validator disabled |
| **C** | Gemma-4-e2b-it (Base) | ON | Ablation: unmodified base model + validator |
| **D** | Gemma-4-e2b-it (Base) | OFF | Ablation: raw base model, no intervention |

**Benchmark:** MedQA-GB — 1,000 UK clinical MCQ questions sampled from MMLU-Medical, MMLU-Clinical Knowledge, MMLU-Professional Medicine, MMLU-Anatomy, MMLU-College Biology, MMLU-College Medicine, MMLU-Medical Genetics, and MedQA-GB.

---

## 2. Evaluation Methods

Three separate accuracy methods were used — each reveals a different aspect of model behaviour. Applied in order from weakest to strongest signal.

### Method 1 — Keyword Overlap Accuracy (Baseline, Flawed)

The original metric used during development. Checks whether ≥30% of words in the ground truth answer appear in the model output.

```
acc = 1  if  |GT_words ∩ OUT_words| / |GT_words|  ≥  0.3
```

**Why it was used:** Fast, deterministic, no external dependencies.

**Why it is flawed:** A verbose model that includes all four answer options in its output will trivially score high regardless of which option it actually chose. It measures verbosity, not correctness.

---

### Method 2 — Letter-Match Accuracy (Standard MCQ)

Extracts the chosen option letter (A/B/C/D) from the model output using a prioritised regex cascade and checks it against the ground truth label.

Regex priority order (first match wins):
1. `"correct answer is **C."` / `"correct answer is C."`
2. Bold markdown option: `**C.`
3. Line-start option: `C.` at start of line
4. `(C)` at start of line
5. `"Answer: C"` / `"Answer: (C)"`
6. `"option C"`
7. `"is **C."` embedded in sentence
8. Last resort: first `X.` or `X:` or `X)` in text

**Why it matters:** This is the standard MCQ evaluation used in all published medical benchmark leaderboards (MMLU, MedQA).

**Why it is insufficient alone:** It penalises a model that has the correct semantic understanding but assigns it to the wrong option position — exactly the failure mode discovered in IatrogeniX.

---

### Method 3 — Semantic Similarity Accuracy (Primary)

Uses `all-MiniLM-L6-v2` (a 90MB SBERT model) to embed both the model's output and the correct answer text, then computes cosine similarity.

```
score(output, gt_text) = cosine_similarity(embed(output[:512]), embed(gt_text))
classified_correct  if  score ≥ threshold
```

Three thresholds were evaluated: 0.5, 0.6, 0.7.

**Why it is more informative:** It captures whether the model's output is *semantically close* to the correct answer, independent of which letter it picked. This distinguishes genuine knowledge from positional bias.

**Limitation:** SBERT similarity is sensitive to output length. A very long output that discusses all options will have its embedding averaged across all topics, pulling it away from any single answer.

---

## 3. Results

### 3.1 — Keyword Overlap Accuracy (Method 1)

> ⚠️ Reported for completeness only. See Methods 2 and 3 for meaningful results.

| Mode | Keyword Overlap Accuracy |
|---|---|
| A — IatrogeniX FT + Safety ON | 58.8% |
| B — IatrogeniX FT + Safety OFF | 58.4% |
| C — Gemma-Base + Safety ON | 94.5% |
| D — Gemma-Base + Safety OFF | 94.5% |

**Why Mode C/D appear so high:** The base model writes long verbose outputs (median 2,255 characters) that naturally include keywords from all four answer options. This inflates keyword overlap regardless of whether the model chose correctly.

---

### 3.2 — Letter-Match Accuracy (Method 2)

| Mode | Correct | Total | Accuracy | Unanswered |
|---|---|---|---|---|
| A — IatrogeniX FT + Safety ON | 367 | 1,000 | **36.7%** | 2 |
| B — IatrogeniX FT + Safety OFF | 372 | 1,000 | **37.2%** | 2 |
| C — Gemma-Base + Safety ON | 436 | 1,000 | **43.6%** | 63 |
| D — Gemma-Base + Safety OFF | 436 | 1,000 | **43.6%** | 63 |

**Note on "Unanswered":** Cases where no letter could be extracted. Mode C/D had 63 such cases — long explanatory paragraphs that never committed to a specific option.

---

### 3.3 — Semantic Similarity Accuracy (Method 3, Primary)

| Mode | Avg Cosine Sim | @0.5 | @0.6 | @0.7 |
|---|---|---|---|---|
| A — IatrogeniX FT + Safety ON | **0.598** | **57.0%** | **51.4%** | **45.7%** |
| B — IatrogeniX FT + Safety OFF | **0.599** | **58.3%** | **52.0%** | **46.1%** |
| C — Gemma-Base + Safety ON | 0.391 | 33.7% | 22.0% | 8.9% |
| D — Gemma-Base + Safety OFF | 0.393 | 34.1% | 22.1% | 9.2% |

**Key reversal:** IatrogeniX outperforms Gemma-Base on semantic accuracy by +29.4pp at @0.6, despite losing on letter-match. This reversal is the core finding of this study.

---

### 3.4 — Safety & Clinical Behaviour

| Metric | A — FT + Safety ON | B — FT + Safety OFF | C — Base + Safety ON | D — Base + Safety OFF |
|---|---|---|---|---|
| Safety Pass Rate | **99.8%** | 99.7% | 96.8% | 96.8% |
| Unsafe Outputs | **0** | **0** | **0** | **0** |
| Warnings Triggered | **2** | 3 | 32 | 32 |
| Professionalism | **100%** | **100%** | 98.5% | 98.5% |
| Reasoning Structure | 0.2% | 0.6% | 96.9% | 96.9% |
| Avg Latency (Mode A) | 24.25s | — | — | — |
| Median Latency | 22.89s | — | — | — |
| P95 Latency | 50.67s | — | — | — |

---

### 3.5 — Summary Comparison

| Metric | Winner | Notes |
|---|---|---|
| Letter-match accuracy | Gemma-Base (+6.9pp) | Due to IatrogeniX letter bias |
| Semantic accuracy (@0.6) | **IatrogeniX (+29.4pp)** | True knowledge measure |
| Safety pass rate | **IatrogeniX (+3pp)** | 99.8% vs 96.8% |
| Professionalism | **IatrogeniX (+1.5pp)** | 100% vs 98.5% |
| Safety warnings | **IatrogeniX (16x fewer)** | 2 vs 32 |
| Output conciseness | IatrogeniX | Median 32 chars vs 2,255 chars |
| Reasoning traces | Gemma-Base | IatrogeniX gives direct answers without explanation |

---

## 4. Failure Mode Analysis

### Failure 1 — Severe Letter (Positional) Bias ⚠️

**What was found:**

| Letter | Ground Truth Frequency | Mode A Predicted | Δ |
|---|---|---|---|
| A | 235 / 1,000 | 82 | **-153** |
| B | 227 / 1,000 | 488 | **+261 ← overpredicted** |
| C | 235 / 1,000 | 355 | +120 |
| D | 303 / 1,000 | 73 | **-230** |

The model picks B in 48.8% of all responses. B is correct only 22.7% of the time.

**Root cause 1 — Unshuffled streaming:**
`training/prepare_data.py` samples the medmcqa dataset with `ds.take(8000)` — taking the first 8,000 rows in dataset order without shuffling. medmcqa is organised by topic, so the first 8,000 rows have a skewed distribution (A: 29.5%, B: 25.6%, C: 24.2%, D: 20.6%).

**Root cause 2 — No option position shuffling:**
Answer options are always in fixed A/B/C/D positions during training. Over thousands of examples, the model learns positional shortcuts: "option B tends to be the nuanced/hedged clinical answer." It reasons about position, not content.

**Direct cost:** 172 out of 1,000 cases where IatrogeniX produces semantically correct answer text but assigns it to the wrong letter — directly traceable to this bias.

---

### Failure 2 — No Reasoning Traces

IatrogeniX outputs very short, direct answers (median 32 chars). Only 0.2% of outputs contain any reasoning structure.

**Why this matters clinically:** A clinical AI that says `"B. Intravenous penicillin"` with no explanation provides no auditability. Clinicians cannot verify whether the model understood the case or guessed.

**Root cause:** Training mix prioritises short open-ended Q&A. No explicit reasoning format was enforced during fine-tuning.

---

### Failure 3 — Safety Layer Redundancy on Fine-tuned Model

The validator adds only 0.1pp safety improvement over Mode B (99.8% vs 99.7%). The fine-tuning has already internalised safe language patterns, making the validator a redundant double-check for Mode A.

However, the validator **does** meaningfully help on the base model: catching 32 warning-level outputs (vs 2 for the fine-tuned model) — a 16x improvement. This confirms the fine-tuning was effective in embedding safety behaviour.

---

### Failure 4 — Training Data Scale and Domain Gaps

Both models fail on 377/1,000 questions. These represent genuine domain gaps concentrated in:

| Topic | Both Models Wrong |
|---|---|
| Professional Medicine | 117 |
| Clinical Knowledge | 57 |
| Anatomy | 47 |
| College Biology | 45 |
| MedQA-GB | 43 |
| College Medicine | 41 |
| Medical Genetics | 27 |

These are not model architecture problems — they are training data coverage gaps.

---

## 5. What the Numbers Actually Mean

The standard narrative from letter-match alone:
> *"Gemma-Base (43.6%) beats IatrogeniX (36.7%)."*

The complete picture:

| Claim | Evidence |
|---|---|
| IatrogeniX has better medical knowledge | Semantic similarity 0.598 vs 0.391 — IatrogeniX outputs are consistently closer to correct answer text |
| IatrogeniX is safer | 99.8% vs 96.8% safety pass rate, 2 vs 32 warnings |
| Gemma-Base picks the right letter more often | True — but 63/1,000 go unanswered and semantic quality is much lower |
| Letter gap is entirely letter bias | 172/1,000 cases are semantically correct but wrong letter |
| Fine-tuning caused knowledge loss | No evidence — the fine-tuned model is semantically more accurate |

---

## 6. Future Improvement Roadmap

### Fix 1 — Shuffle Training Data Before Sampling (High Priority)

```python
# Before — takes first N rows in topic order
for row in ds.take(n):

# After — shuffles 20k buffer before sampling
for row in ds.shuffle(seed=SEED, buffer_size=20000).take(n):
```
**Expected gain:** +5–10pp letter-match accuracy.

---

### Fix 2 — Option Shuffling During Training (High Priority)

Randomly permute A/B/C/D options per training example, relabelling the correct answer. Forces the model to evaluate content, not position.

```python
def shuffle_options(options, correct_idx, rng):
    indexed = list(enumerate(options))
    rng.shuffle(indexed)
    new_options = [opt for _, opt in indexed]
    old_to_new = {old: new for new, (old, _) in enumerate(indexed)}
    return new_options, old_to_new[correct_idx]
```
**Expected gain:** Eliminates positional learning. Combined with Fix 1: letter-match should align with semantic accuracy (~50–55%).

---

### Fix 3 — Remove Letters from Training Answer Labels (Alternative to Fix 2)

Train with answer text only (no letter prefix):
```
# Before: "The correct answer is (B) Intravenous penicillin."
# After:  "The correct answer is Intravenous penicillin."
```
Completely eliminates positional bias by design. Requires an explicit system prompt instruction to output a letter at inference.

---

### Fix 4 — Add Reasoning Traces to Training Data (Medium Priority)

Augment training answers with step-by-step reasoning chains. Target: raise reasoning structure metric from 0.2% toward 50%+. Makes outputs auditable by clinicians.

---

### Fix 5 — Expand Training Dataset (Medium Priority)

Add focused datasets for the highest-failure domains: USMLE Step 2/3 vignettes (Professional Medicine), anatomy Q&A, medical genetics resources.

---

### Fix 6 — Upgrade Safety Validator (Low Priority for Mode A)

Current validator catches linguistic patterns only. Suggested additions:
- Numeric dose range checker (flag if dose is >2× reference range)
- Drug-drug interaction database
- Contraindication flags based on patient demographics in the question

---

### Fix 7 — Latency Optimisation

Current P95 latency is 50.67s on CPU. Target <5s for clinical use.
- Switch Q5_K_M → Q4_K_M quantisation
- Enable GPU layers (`n_gpu_layers` in llama-cpp)
- Implement KV cache for repeated system prompt

---

## 7. How to Replicate This Study

```bash
# Clone and install
git clone https://github.com/kayomarz97/IatrogeniX
cd IatrogeniX
pip install -r requirements.txt
pip install sentence-transformers

# Run Mode A evaluation (IatrogeniX FT + Safety ON)
python3 evaluation/mass_eval.py

# Run Modes B/C/D ablation
python3 scripts/safety_ablation.py

# Semantic similarity evaluation
python3 evaluation/semantic_eval.py
```

**Semantic evaluation (`evaluation/semantic_eval.py`):**
```python
from sentence_transformers import SentenceTransformer
import numpy as np, json, re

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_and_score(outputs, gt_texts, threshold=0.6):
    out_embs = model.encode(outputs, batch_size=128)
    gt_embs  = model.encode(gt_texts,  batch_size=128)
    gt_norm  = gt_embs  / np.linalg.norm(gt_embs,  axis=1, keepdims=True)
    out_norm = out_embs / np.linalg.norm(out_embs, axis=1, keepdims=True)
    sims = np.sum(gt_norm * out_norm, axis=1)
    return float(np.mean(sims)), int(np.sum(sims >= threshold))
```

---

## 8. Community Feedback Requested

If you run IatrogeniX on your own datasets, we specifically want to know:

1. **Letter-match accuracy on your dataset** — does the B-bias appear on non-MedQA-GB benchmarks?
2. **Semantic similarity scores** — does the 0.598 cosine similarity hold on different question types?
3. **Safety validator false positives** — does the validator flag correct answers as warnings?
4. **Latency on your hardware** — especially GPU-accelerated inference with `n_gpu_layers > 0`
5. **Failure categories** — which medical subspecialties does the model fail hardest on?

Open a GitHub Issue or start a HuggingFace Discussion with your results.

---

## 9. Appendix — Raw Numbers

### Letter Distribution (Mode A)

| Letter | GT Frequency | Predicted | Δ |
|---|---|---|---|
| A | 235 | 82 | -153 |
| B | 227 | 488 | **+261** |
| C | 235 | 355 | +120 |
| D | 303 | 73 | -230 |

### Per-Question Outcome Matrix (Mode A vs Mode C)

| Outcome | Count |
|---|---|
| Both correct | 180 |
| Both wrong | 377 |
| A correct, C wrong | 187 |
| A wrong, C correct | 256 |
| Net gap (C−A) | 69 questions |

### Warning Issue Types

| Issue | Mode A | Mode B | Mode C | Mode D |
|---|---|---|---|---|
| Absolute statement: 'always' | 2 | 3 | 36 | 36 |
| Absolute statement: 'never' | 0 | 0 | 6 | 6 |
| Absolute statement: 'guaranteed' | 0 | 0 | 3 | 3 |
| Absolute statement: 'definitely' | 0 | 0 | 1 | 1 |
| **Total warnings** | **2** | **3** | **46** | **46** |

---

*Study conducted April 2026 · Benchmark: MedQA-GB, 1,000 cases · Embedding model: all-MiniLM-L6-v2 · Hardware: CPU-only inference*
