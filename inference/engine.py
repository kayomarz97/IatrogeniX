"""
IatrogeniX — inference/engine.py
==================================
FastAPI inference server. Loads the fine-tuned GGUF model via llama-cpp-python
and exposes two endpoints:
  POST /generate       — raw model output
  POST /generate/safe  — model output + safety validation layer

Run on VPS (8 GB RAM, no GPU needed for 2B model):
  uvicorn inference.engine:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os, time, json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Config ────────────────────────────────────────────────────────────────────
GGUF_PATH   = os.environ.get("IATROGENIX_MODEL", "models/iatrogenix-q4_k_m.gguf")
N_CTX       = int(os.environ.get("N_CTX", "4096"))
N_GPU_LAYERS= int(os.environ.get("N_GPU_LAYERS", "0"))   # 0 = CPU-only (VPS)
MAX_TOKENS_DEFAULT = 512

SYSTEM_PROMPT = (
    "You are a clinical reasoning assistant. Provide evidence-based answers "
    "with step-by-step reasoning. Always mention drug names with doses where applicable."
)

# Gemma 4 chat template tokens (for llama-cpp-python manual formatting)
# Note: Unsloth's tokenizer handles this at training time.
# For llama.cpp inference we must build the prompt manually.
GEMMA4_TURN_START = "<|turn>"
GEMMA4_TURN_END   = "<turn|>"

def build_gemma4_prompt(question: str, system: str = SYSTEM_PROMPT) -> str:
    return (
        f"{GEMMA4_TURN_START}system\n{system}{GEMMA4_TURN_END}"
        f"{GEMMA4_TURN_START}user\n{question}{GEMMA4_TURN_END}"
        f"{GEMMA4_TURN_START}model\n"
    )

# ── App & Model ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="IatrogeniX Inference API",
    description=(
        "Hybrid clinical LLM — fine-tuned Gemma 4 E2B-it (Q5_K_M GGUF) "
        "with safety validation layer. Portfolio/demo project. NOT for clinical use."
    ),
    version="1.0.0",
)

_llm = None
_validator = None

def get_llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama
        model_path = Path(GGUF_PATH)
        if not model_path.exists():
            raise RuntimeError(
                f"Model not found at {GGUF_PATH}. "
                "Run training/train.py on Colab and copy the GGUF file here."
            )
        print(f"Loading GGUF model: {model_path} ...")
        _llm = Llama(
            model_path   = str(model_path),
            n_ctx        = N_CTX,
            n_gpu_layers = N_GPU_LAYERS,
            verbose      = False,
        )
        print("Model loaded.")
    return _llm

def get_validator():
    global _validator
    if _validator is None:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from safety.validator import SafetyValidator
        _validator = SafetyValidator()
    return _validator

# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    question: str = Field(..., description="Clinical question to answer")
    system_prompt: Optional[str] = Field(None, description="Override system prompt")
    temperature: float = Field(0.3, ge=0.0, le=1.0)
    top_p: float       = Field(0.9, ge=0.0, le=1.0)
    max_tokens: int    = Field(MAX_TOKENS_DEFAULT, ge=1, le=2048)
    ground_truth: Optional[str] = Field(None, description="For safety validation comparison (eval use)")

class ModelOutput(BaseModel):
    question: str
    model_output: str
    generation_time_seconds: float
    tokens_generated: int
    model_path: str

class SafeModelOutput(ModelOutput):
    safety_status: str
    safety_issues: list[dict]
    hallucinations: list[str]
    dose_errors: list[str]
    overconfident: list[str]
    corrected_output: str

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": GGUF_PATH, "disclaimer": "NOT for clinical use"}

@app.post("/generate", response_model=ModelOutput)
def generate(req: GenerateRequest):
    """Raw model output — no safety filtering."""
    llm    = get_llm()
    prompt = build_gemma4_prompt(req.question, req.system_prompt or SYSTEM_PROMPT)

    t0 = time.time()
    try:
        result = llm(
            prompt,
            max_tokens  = req.max_tokens,
            temperature = req.temperature,
            top_p       = req.top_p,
            stop        = [GEMMA4_TURN_START, GEMMA4_TURN_END, "<eos>"],
            echo        = False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    elapsed  = time.time() - t0
    text     = result["choices"][0]["text"].strip()
    n_tokens = result["usage"]["completion_tokens"]

    return ModelOutput(
        question               = req.question,
        model_output           = text,
        generation_time_seconds= round(elapsed, 2),
        tokens_generated       = n_tokens,
        model_path             = GGUF_PATH,
    )

@app.post("/generate/safe", response_model=SafeModelOutput)
def generate_safe(req: GenerateRequest):
    """Model output + safety validation layer."""
    raw      = generate(req)
    validator= get_validator()
    v_result = validator.validate(
        raw.model_output,
        ground_truth = req.ground_truth,
        question     = req.question,
    )

    return SafeModelOutput(
        **raw.model_dump(),
        safety_status    = v_result.status,
        safety_issues    = [i.to_dict() if hasattr(i, "to_dict") else
                            {"type": i.issue_type, "severity": i.severity,
                             "description": i.description}
                            for i in v_result.issues],
        hallucinations   = v_result.hallucinations,
        dose_errors      = v_result.dose_errors,
        overconfident    = v_result.overconfident,
        corrected_output = v_result.corrected_text,
    )

@app.get("/model/info")
def model_info():
    """Return model metadata."""
    p = Path(GGUF_PATH)
    return {
        "model_file": GGUF_PATH,
        "exists": p.exists(),
        "size_gb": round(p.stat().st_size / 1e9, 2) if p.exists() else None,
        "base_model": "unsloth/gemma-4-E2B-it",
        "fine_tuning": "LoRA r=16, medical Q&A",
        "quantisation": "Q5_K_M",
        "disclaimer": "Portfolio/demo project. NOT for clinical use.",
    }

# ── CLI standalone (non-server) inference ────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from safety.validator import SafetyValidator

    parser = argparse.ArgumentParser(description="IatrogeniX CLI inference")
    parser.add_argument("--question", "-q", required=True, help="Clinical question")
    parser.add_argument("--safe", action="store_true", help="Enable safety layer")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    llm    = get_llm()
    prompt = build_gemma4_prompt(args.question)
    print(f"\nQuestion: {args.question}\n{'='*60}")

    t0 = time.time()
    result = llm(prompt, max_tokens=args.max_tokens, temperature=args.temperature,
                 stop=[GEMMA4_TURN_START, GEMMA4_TURN_END, "<eos>"], echo=False)
    elapsed = time.time() - t0
    text    = result["choices"][0]["text"].strip()

    print(f"Answer ({elapsed:.1f}s):\n{text}")

    if args.safe:
        v = SafetyValidator()
        vr = v.validate(text, question=args.question)
        print(f"\n[Safety: {vr.status.upper()}] issues={len(vr.issues)}")
        for issue in vr.issues:
            print(f"  [{issue.severity}] {issue.description}")
