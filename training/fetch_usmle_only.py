import json
from datasets import load_dataset

def make_entry(question: str, answer: str, source: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": "You are a clinical reasoning assistant. Provide evidence-based answers with step-by-step reasoning. Always mention drug names with doses where applicable."},
            {"role": "user",      "content": question.strip()},
            {"role": "assistant", "content": answer.strip()},
        ],
        "source": source,
    }

def main():
    print("Fetching USMLE dataset...", flush=True)
    ds = load_dataset("medalpaca/medical_meadow_medqa", split="train", streaming=True)
    entries = []
    
    for row in ds.take(12000):
        instruction = str(row.get("instruction", "")).strip()
        input_text  = str(row.get("input", "")).strip()
        q = f"{instruction}\n{input_text}".strip()
        a = str(row.get("output", "")).strip()
        
        if not q or not a:
            continue
        entries.append(make_entry(q, a, "medqa-usmle"))

    print(f"Loaded {len(entries)} USMLE questions. Appending to dataset...", flush=True)
    
    # Append to the existing dataset
    with open("training/data/processed_dataset.jsonl", "a") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print("✅ Done! You can now run train.py!")

if __name__ == "__main__":
    main()
