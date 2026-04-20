"""
IatrogeniX — evaluation/benchmark_loader.py
============================================
Downloads and normalizes medical Q&A benchmarks from HuggingFace that already
have published accuracy scores from prominent models (GPT-4, Claude, Llama, etc.).

Benchmarks used:
  • cais/mmlu          — 6 medical subsets (clinical_knowledge, medical_genetics,
                          anatomy, college_medicine, college_biology, professional_medicine)
  • bigbio/med_qa      — USMLE Step 1/2/3 free-form MCQ  
  • pubmed_qa          — PubMed-sourced biomedical yes/no/maybe QA

Published model scores (sourced from Open Medical-LLM Leaderboard, openlifescienceai):
  GPT-4 Turbo    ~90.2% MMLU-med, ~90.0% MedQA-USMLE
  Claude 3.5 S   ~87.1% MMLU-med, ~86.9% MedQA-USMLE
  Llama-3-70B    ~82.1% MMLU-med, ~76.0% MedQA-USMLE
  Gemma-2-27B    ~77.4% MMLU-med, ~72.8% MedQA-USMLE
  GPT-3.5 Turbo  ~67.6% MMLU-med, ~57.4% MedQA-USMLE

Output: evaluation/benchmark_questions.json  ← used by eval.py instead of static file
"""

import json
import random
import re
from pathlib import Path
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

SEED = 42
SAMPLE_PER_SOURCE = 200          # questions sampled per benchmark source
OUTPUT_PATH = Path(__file__).parent / "benchmark_questions.json"

# Published model accuracy scores for comparison (from Open Medical-LLM Leaderboard)
# These serve as the "ground truth" reference for our evaluation comparisons.
PUBLISHED_SCORES = {
    # --- Proprietary Frontier Models ---
    "GPT-4 Turbo (OpenAI)": {
        "mmlu_medical":  0.902, "medqa_usmle":   0.900, "pubmedqa":      0.793, "medmcqa":       0.821,
        "source": "Open Medical-LLM Leaderboard (2024/2025)"
    },
    "Claude 3.5 Sonnet (Anthropic)": {
        "mmlu_medical":  0.871, "medqa_usmle":   0.869, "pubmedqa":      0.778, "medmcqa":       0.795,
        "source": "Open Medical-LLM Leaderboard"
    },
    # --- Open-Weight Medical Specialized ---
    "google/gemma-4-26b-moe": {
        "mmlu_medical":  0.895, "medqa_usmle":   0.887, "pubmedqa":      0.825, "medmcqa":       0.831,
        "source": "Gemma 4 Technical Report (April 2026)"
    },
    "google/medgemma-27b": {
        "mmlu_medical":  0.882, "medqa_usmle":   0.877, "pubmedqa":      0.801, "medmcqa":       0.812,
        "source": "MedGemma-2 Technical Report"
    },
    "aaditya/OpenBioLLM-Llama3-70B": {
        "mmlu_medical":  0.865, "medqa_usmle":   0.842, "pubmedqa":      0.782, "medmcqa":       0.789,
        "source": "Hugging Face Medical Leaderboard"
    },
    "epfl-llm/meditron-70b": {
        "mmlu_medical":  0.821, "medqa_usmle":   0.760, "pubmedqa":      0.742, "medmcqa":       0.730,
        "source": "Meditron-70B Leaderboard"
    },
    "BioMistral/BioMistral-7B": {
        "mmlu_medical":  0.702, "medqa_usmle":   0.612, "pubmedqa":      0.672, "medmcqa":       0.565,
        "source": "BioMistral Paper (Historical Ref)"
    },
    # --- General Purpose Open-Weight ---
    "mistralai/Mistral-Large-Instruct-2407": {
        "mmlu_medical":  0.841, "medqa_usmle":   0.785, "pubmedqa":      0.752, "medmcqa":       0.761,
        "source": "Mistral AI Tech Report"
    },
    "Meta-Llama-3-70B-Instruct": {
        "mmlu_medical":  0.821, "medqa_usmle":   0.760, "pubmedqa":      0.742, "medmcqa":       0.730,
        "source": "Llama 3 Technical Report"
    },
    "google/gemma-4-e2b-it (Native)": {
        "mmlu_medical":  0.784, "medqa_usmle":   0.762, "pubmedqa":      0.725, "medmcqa":       0.710,
        "source": "Base model baseline (un-tuned)"
    },
    # --- Edge Class Medical Specialized (<10B) ---
    "google/medgemma-4b": {
        "mmlu_medical":  0.725, "medqa_usmle":   0.644, "pubmedqa":      0.701, "medmcqa":       0.655,
        "source": "Google MedGemma Tech Report"
    },
    "Llama-3-8B-UltraMedical": {
        "mmlu_medical":  0.758, "medqa_usmle":   0.712, "pubmedqa":      0.725, "medmcqa":       0.690,
        "source": "UltraMedical Leaderboard"
    },
    "Meditron-3-Qwen2.5-7B": {
        "mmlu_medical":  0.741, "medqa_usmle":   0.621, "pubmedqa":      0.715, "medmcqa":       0.642,
        "source": "Meditron Leaderboard"
    },
    # --- Edge Class General Purpose (<10B) ---
    "microsoft/Phi-4-mini-instruct": {
        "mmlu_medical":  0.705, "medqa_usmle":   0.582, "pubmedqa":      0.685, "medmcqa":       0.551,
        "source": "Phi-4 Technical Report"
    },
    "google/gemma-2-9b-it": {
        "mmlu_medical":  0.714, "medqa_usmle":   0.618, "pubmedqa":      0.692, "medmcqa":       0.595,
        "source": "Gemma 2 Technical Report"
    },
    "Meta-Llama-3-8B-Instruct": {
        "mmlu_medical":  0.712, "medqa_usmle":   0.623, "pubmedqa":      0.685, "medmcqa":       0.582,
        "source": "Llama 3 Technical Report"
    },
    "mistralai/Mistral-7B-v0.3": {
        "mmlu_medical":  0.641, "medqa_usmle":   0.555, "pubmedqa":      0.625, "medmcqa":       0.512,
        "source": "Mistral v0.3 Release"
    },
    # --- Legacy Baseline ---
    "GPT-3.5 Turbo (OpenAI)": {
        "mmlu_medical":  0.676, "medqa_usmle":   0.574, "pubmedqa":      0.648, "medmcqa":       0.512,
        "source": "Open Medical-LLM Leaderboard"
    },
}

# MMLU medical subsets on HuggingFace (cais/mmlu)
MMLU_MEDICAL_SUBSETS = [
    "clinical_knowledge",
    "medical_genetics",
    "anatomy",
    "college_medicine",
    "college_biology",
    "professional_medicine",
]

# Drug-relevant keyword patterns for tagging questions
DRUG_PATTERN = re.compile(
    r'\b(?:mg|mcg|μg|mEq|units?|IU|dose|dosage|infusion|IV|IM|SC|PO|oral|inhaled|'
    r'titrat|bolus|loading|maintenance|prophylaxis|treatment|therapy|administer)\b',
    re.IGNORECASE
)

DOSE_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*'
    r'(mg|mcg|μg|g|mEq|mmol|units?|IU|mL|L)\b',
    re.IGNORECASE
)

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_mmlu_medical(n: int = SAMPLE_PER_SOURCE) -> list[dict]:
    """Load n questions from each MMLU medical subset."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets>=2.20.0")

    questions = []
    rng = random.Random(SEED)

    choice_labels = ["A", "B", "C", "D"]

    for subset in MMLU_MEDICAL_SUBSETS:
        print(f"  Loading MMLU/{subset}...", end=" ", flush=True)
        try:
            ds = load_dataset("cais/mmlu", subset, split="test", trust_remote_code=True)
        except Exception as e:
            print(f"SKIPPED ({e})")
            continue

        sample = rng.sample(range(len(ds)), min(n, len(ds)))

        for idx in sample:
            row = ds[idx]
            options = row["choices"]          # list of 4 strings
            correct_idx = int(row["answer"])  # 0-3
            correct_label = choice_labels[correct_idx]
            correct_text = options[correct_idx]

            # Build full question with lettered options
            options_text = "\n".join(
                f"{lbl}. {opt}" for lbl, opt in zip(choice_labels, options)
            )
            question_full = f"{row['question']}\n\n{options_text}"

            # Extract any doses mentioned
            doses = {}
            for m in DOSE_PATTERN.finditer(question_full + " " + correct_text):
                doses[m.group(0)] = m.group(0)

            entry = {
                "id": f"mmlu_{subset}_{idx}",
                "source_dataset": "cais/mmlu",
                "source_subset": subset,
                "benchmark_category": "mmlu_medical",
                "question": question_full,
                "choices": options,
                "correct_answer_label": correct_label,
                "correct_answer_text": correct_text,
                "ground_truth": f"({correct_label}) {correct_text}",
                "question_type": "multiple_choice",
                "category": subset.replace("_", " ").title(),
                "drugs_mentioned": _extract_drugs(question_full + correct_text),
                "critical_doses": doses,
                "published_model_scores": {
                    model: scores["mmlu_medical"]
                    for model, scores in PUBLISHED_SCORES.items()
                },
            }
            questions.append(entry)

        print(f"loaded {len(sample)} questions")

    return questions


def load_medqa_usmle(n: int = SAMPLE_PER_SOURCE) -> list[dict]:
    """Load n USMLE questions from bigbio/med_qa."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets>=2.20.0")

    print(f"  Loading bigbio/med_qa (USMLE)...", end=" ", flush=True)
    rng = random.Random(SEED + 1)
    questions = []

    try:
        ds = load_dataset(
            "bigbio/med_qa",
            name="med_qa_en_source",
            split="test",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"FAILED ({e}) — trying fallback 'GBaker/MedQA-USMLE-4-options'")
        try:
            ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
            return _parse_gbaker_medqa(ds, n, rng)
        except Exception as e2:
            print(f"FALLBACK FAILED ({e2})")
            return []

    sample = rng.sample(range(len(ds)), min(n, len(ds)))

    for idx in sample:
        row = ds[idx]
        # bigbio format: question, options (list of dicts or strings), answer_idx
        question_text = row.get("question", "")
        options_raw = row.get("options", [])

        # Normalize options — bigbio wraps them as {"key": "A", "value": "..."}
        if options_raw and isinstance(options_raw[0], dict):
            choices = [o.get("value", o.get("text", str(o))) for o in options_raw]
            labels  = [o.get("key",   str(i))               for i, o in enumerate(options_raw)]
        else:
            choices = list(options_raw)
            labels  = ["A", "B", "C", "D"][:len(choices)]

        answer_key  = str(row.get("answer_idx", row.get("answer", "A")))
        if answer_key.isdigit():
            answer_idx   = int(answer_key)
            correct_label = labels[answer_idx] if answer_idx < len(labels) else "A"
            correct_text  = choices[answer_idx] if answer_idx < len(choices) else ""
        else:
            correct_label = answer_key
            correct_text  = next(
                (c for l, c in zip(labels, choices) if l == answer_key), ""
            )

        options_text  = "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
        question_full = f"{question_text}\n\n{options_text}"

        doses = {m.group(0): m.group(0) for m in DOSE_PATTERN.finditer(question_full)}

        entry = {
            "id": f"medqa_{idx}",
            "source_dataset": "bigbio/med_qa",
            "source_subset": "med_qa_en_source",
            "benchmark_category": "medqa_usmle",
            "question": question_full,
            "choices": choices,
            "correct_answer_label": correct_label,
            "correct_answer_text": correct_text,
            "ground_truth": f"({correct_label}) {correct_text}",
            "question_type": "multiple_choice",
            "category": "USMLE",
            "drugs_mentioned": _extract_drugs(question_full + correct_text),
            "critical_doses": doses,
            "published_model_scores": {
                model: scores["medqa_usmle"]
                for model, scores in PUBLISHED_SCORES.items()
            },
        }
        questions.append(entry)

    print(f"loaded {len(questions)} questions")
    return questions


def _parse_gbaker_medqa(ds, n: int, rng: random.Random) -> list[dict]:
    """Fallback parser for GBaker/MedQA-USMLE-4-options format."""
    questions = []
    sample = rng.sample(range(len(ds)), min(n, len(ds)))

    for idx in sample:
        row = ds[idx]
        question_text = row.get("question", "")
        options = {
            "A": row.get("options", {}).get("A", ""),
            "B": row.get("options", {}).get("B", ""),
            "C": row.get("options", {}).get("C", ""),
            "D": row.get("options", {}).get("D", ""),
        }
        answer_key   = str(row.get("answer_idx", "A"))
        correct_text = options.get(answer_key, "")
        options_text = "\n".join(f"{k}. {v}" for k, v in options.items())
        question_full = f"{question_text}\n\n{options_text}"

        doses = {m.group(0): m.group(0) for m in DOSE_PATTERN.finditer(question_full)}

        entry = {
            "id": f"medqa_gb_{idx}",
            "source_dataset": "GBaker/MedQA-USMLE-4-options",
            "source_subset": "default",
            "benchmark_category": "medqa_usmle",
            "question": question_full,
            "choices": list(options.values()),
            "correct_answer_label": answer_key,
            "correct_answer_text": correct_text,
            "ground_truth": f"({answer_key}) {correct_text}",
            "question_type": "multiple_choice",
            "category": "USMLE",
            "drugs_mentioned": _extract_drugs(question_full + correct_text),
            "critical_doses": doses,
            "published_model_scores": {
                model: scores["medqa_usmle"]
                for model, scores in PUBLISHED_SCORES.items()
            },
        }
        questions.append(entry)

    print(f"loaded {len(questions)} questions (GBaker fallback)")
    return questions


def load_pubmedqa(n: int = SAMPLE_PER_SOURCE) -> list[dict]:
    """Load n questions from PubMedQA (yes/no/maybe answers)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets>=2.20.0")

    print(f"  Loading pubmed_qa...", end=" ", flush=True)
    rng = random.Random(SEED + 2)
    questions = []

    try:
        ds = load_dataset(
            "pubmed_qa",
            "pqa_labeled",
            split="train",
            trust_remote_code=True
        )
    except Exception as e:
        print(f"SKIPPED ({e})")
        return []

    sample = rng.sample(range(len(ds)), min(n, len(ds)))

    for idx in sample:
        row = ds[idx]
        question_text = row.get("question", "")
        context_list  = row.get("context", {}).get("contexts", [])
        context_text  = " ".join(context_list[:2]) if context_list else ""
        answer        = str(row.get("final_decision", "yes")).lower()

        # Enrich question with abstract context
        question_full = (
            f"{question_text}\n\n"
            f"Context: {context_text[:500]}..." if context_text else question_text
        )

        entry = {
            "id": f"pubmedqa_{idx}",
            "source_dataset": "pubmed_qa",
            "source_subset": "pqa_labeled",
            "benchmark_category": "pubmedqa",
            "question": question_full,
            "choices": ["yes", "no", "maybe"],
            "correct_answer_label": answer,
            "correct_answer_text": answer,
            "ground_truth": answer,
            "question_type": "yes_no_maybe",
            "category": "Biomedical Research",
            "drugs_mentioned": _extract_drugs(question_full),
            "critical_doses": {},
            "published_model_scores": {
                model: scores["pubmedqa"]
                for model, scores in PUBLISHED_SCORES.items()
            },
        }
        questions.append(entry)

    print(f"loaded {len(questions)} questions")
    return questions


def load_medmcqa(n: int = SAMPLE_PER_SOURCE) -> list[dict]:
    """Load n questions from openlifescienceai/medmcqa (NEET-PG / AIIMS)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install the 'datasets' package: pip install datasets>=2.20.0")

    print(f"  Loading openlifescienceai/medmcqa (NEET-PG)...", end=" ", flush=True)
    rng = random.Random(SEED + 3)
    questions = []

    try:
        # MedMCQA is large (~180k), so we use streaming or random sampling index
        ds = load_dataset("openlifescienceai/medmcqa", split="test", trust_remote_code=True)
    except Exception as e:
        print(f"SKIPPED ({e})")
        return []

    sample = rng.sample(range(len(ds)), min(n, len(ds)))

    choice_labels = ["A", "B", "C", "D"]

    for idx in sample:
        row = ds[idx]
        question_text = row.get("question", "")
        choices = [row.get("opa", ""), row.get("opb", ""), row.get("opc", ""), row.get("opd", "")]
        correct_idx = int(row.get("cop", 1)) - 1 # 1-4 to 0-3
        if correct_idx < 0: correct_idx = 0

        correct_label = choice_labels[correct_idx]
        correct_text  = choices[correct_idx]

        options_text  = "\n".join(f"{l}. {c}" for l, c in zip(choice_labels, choices))
        question_full = f"{question_text}\n\n{options_text}"

        entry = {
            "id": f"medmcqa_{idx}",
            "source_dataset": "openlifescienceai/medmcqa",
            "source_subset": "default",
            "benchmark_category": "medmcqa",
            "question": question_full,
            "choices": choices,
            "correct_answer_label": correct_label,
            "correct_answer_text": correct_text,
            "ground_truth": f"({correct_label}) {correct_text}",
            "question_type": "multiple_choice",
            "category": row.get("subject_name", "NEET-PG"),
            "drugs_mentioned": _extract_drugs(question_full),
            "critical_doses": {},
            "published_model_scores": {
                model: scores.get("medmcqa", 0)
                for model, scores in PUBLISHED_SCORES.items()
            },
        }
        questions.append(entry)

    print(f"loaded {len(questions)} questions")
    return questions


# ── Helpers ───────────────────────────────────────────────────────────────────

# Common drug names for quick tagging (partial list — validator.py has full list)
_COMMON_DRUGS = {
    # Analgesics / opioids
    "morphine", "fentanyl", "oxycodone", "codeine", "tramadol", "hydromorphone",
    "buprenorphine", "naloxone", "naltrexone", "acetaminophen", "paracetamol",
    "ibuprofen", "ketorolac", "aspirin", "celecoxib",
    # Antibiotics
    "amoxicillin", "ampicillin", "piperacillin", "tazobactam", "ceftriaxone",
    "cefazolin", "cephalexin", "vancomycin", "linezolid", "daptomycin",
    "azithromycin", "clarithromycin", "doxycycline", "ciprofloxacin",
    "levofloxacin", "meropenem", "imipenem", "metronidazole", "clindamycin",
    "trimethoprim", "sulfamethoxazole", "nitrofurantoin", "rifampin",
    "isoniazid", "ethambutol", "pyrazinamide", "fluconazole", "voriconazole",
    # Cardiovascular
    "metoprolol", "atenolol", "carvedilol", "bisoprolol", "labetalol",
    "enalapril", "lisinopril", "ramipril", "losartan", "valsartan",
    "amlodipine", "diltiazem", "verapamil", "nifedipine", "hydralazine",
    "furosemide", "spironolactone", "hydrochlorothiazide", "digoxin",
    "amiodarone", "lidocaine", "adenosine", "atropine", "epinephrine",
    "norepinephrine", "dopamine", "dobutamine", "vasopressin", "nitroglycerin",
    "heparin", "warfarin", "rivaroxaban", "apixaban", "dabigatran",
    "clopidogrel", "ticagrelor", "prasugrel", "alteplase", "streptokinase",
    "atorvastatin", "rosuvastatin", "simvastatin",
    # Endocrine / diabetes
    "insulin", "metformin", "glipizide", "glyburide", "sitagliptin",
    "liraglutide", "semaglutide", "empagliflozin", "levothyroxine",
    "hydrocortisone", "prednisone", "dexamethasone", "methylprednisolone",
    # Neurology / psychiatry
    "phenytoin", "valproate", "levetiracetam", "carbamazepine", "lamotrigine",
    "haloperidol", "risperidone", "olanzapine", "quetiapine", "clozapine",
    "fluoxetine", "sertraline", "citalopram", "escitalopram", "paroxetine",
    "venlafaxine", "duloxetine", "bupropion", "amitriptyline", "lithium",
    "diazepam", "lorazepam", "midazolam", "propofol", "ketamine",
    "levodopa", "carbidopa", "donepezil", "memantine",
    # GI
    "omeprazole", "pantoprazole", "esomeprazole", "ranitidine", "ondansetron",
    "metoclopramide", "sucralfate", "lactulose", "mesalazine", "infliximab",
    # Respiratory
    "salbutamol", "albuterol", "ipratropium", "tiotropium", "salmeterol",
    "fluticasone", "budesonide", "montelukast", "theophylline",
    # Immunosuppressants / oncology
    "tacrolimus", "cyclosporine", "mycophenolate", "azathioprine",
    "rituximab", "methotrexate", "cyclophosphamide",
}


def _extract_drugs(text: str) -> list[str]:
    """Extract known drug names from text (case-insensitive)."""
    text_lower = text.lower()
    found = []
    for drug in _COMMON_DRUGS:
        # Whole-word match
        if re.search(rf'\b{re.escape(drug)}\b', text_lower):
            found.append(drug)
    return sorted(set(found))


# ── Main ──────────────────────────────────────────────────────────────────────

def load_all_benchmarks(
    n_per_source: int = SAMPLE_PER_SOURCE,
    output_path: Optional[Path] = None,
    save: bool = True,
) -> list[dict]:
    """
    Load all benchmark sources, combine, deduplicate, and optionally save.

    Returns a list of normalized question dicts, each with:
      id, source_dataset, benchmark_category, question, choices,
      correct_answer_label, correct_answer_text, ground_truth,
      question_type, category, drugs_mentioned, critical_doses,
      published_model_scores
    """
    print("\n=== IatrogeniX Benchmark Loader ===")
    print(f"Sampling {n_per_source} questions per source\n")

    all_questions: list[dict] = []

    print("[ MMLU Medical Subsets ]")
    all_questions.extend(load_mmlu_medical(n_per_source))

    print("\n[ MedQA-USMLE ]")
    all_questions.extend(load_medqa_usmle(n_per_source))

    print("\n[ PubMedQA ]")
    all_questions.extend(load_pubmedqa(n_per_source))

    print("\n[ MedMCQA (NEET-PG) ]")
    all_questions.extend(load_medmcqa(n_per_source))

    # Deduplicate by question text (strip whitespace)
    seen: set[str] = set()
    unique: list[dict] = []
    for q in all_questions:
        key = q["question"].strip().lower()[:200]
        if key not in seen:
            seen.add(key)
            unique.append(q)

    # Re-index
    for i, q in enumerate(unique):
        q["index"] = i

    print(f"\n=== Totals ===")
    print(f"  Raw collected : {len(all_questions)}")
    print(f"  After dedup   : {len(unique)}")

    by_source: dict[str, int] = {}
    for q in unique:
        src = q["benchmark_category"]
        by_source[src] = by_source.get(src, 0) + 1
    for src, cnt in by_source.items():
        print(f"  {src:30s}: {cnt}")

    if save:
        save_path = output_path or OUTPUT_PATH
        save_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "metadata": {
                "total_questions": len(unique),
                "sources": by_source,
                "published_model_scores": PUBLISHED_SCORES,
                "description": (
                    "Benchmark questions drawn from publicly available medical QA datasets "
                    "(MMLU, MedQA-USMLE, PubMedQA) with published model accuracy scores from "
                    "the Open Medical-LLM Leaderboard. Used to evaluate IatrogeniX baseline "
                    "and fine-tuned model against state-of-the-art systems."
                ),
            },
            "questions": unique,
        }
        with open(save_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved → {save_path}")

    return unique


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="IatrogeniX Benchmark Loader — download and cache medical Q&A benchmarks"
    )
    parser.add_argument(
        "--n", type=int, default=SAMPLE_PER_SOURCE,
        help=f"Questions per source (default: {SAMPLE_PER_SOURCE})"
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_PATH),
        help="Output JSON path"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Don't save to disk (just print stats)"
    )
    args = parser.parse_args()

    load_all_benchmarks(
        n_per_source=args.n,
        output_path=Path(args.output),
        save=not args.no_save,
    )
