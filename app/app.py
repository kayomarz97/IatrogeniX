import sys
import os
from pathlib import Path
import time

# Add root to sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import gradio as gr
from llama_cpp import Llama
from safety.validator import SafetyValidator

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("IATROGENIX_MODEL", "models/iatrogenix-q5_k_m.gguf")
GGUF_URL = "https://huggingface.co/your-username/IatrogeniX/resolve/main/models/iatrogenix-q5_k_m.gguf"

# ── Model Loader ──────────────────────────────────────────────────────────────
_llm = None
def get_llm():
    global _llm
    if _llm is None:
        if not Path(MODEL_PATH).exists():
            # In a real HF Space, we'd download if missing, 
            # but usually the file is synced with the repo.
            return None
        _llm = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=2048,
            n_gpu_layers=0, # CPU by default on basic Spaces
            verbose=False
        )
    return _llm

# ── Safety Logic ──────────────────────────────────────────────────────────────
validator = SafetyValidator()

def generate_safe_response(question, temperature, max_tokens, mode):
    llm = get_llm()
    if llm is None:
        return "Error: Model file not found. Please ensure the GGUF is in the 'models/' directory.", [], "Error"

    # Build Gemma 4 prompt
    prompt = f"<|turn|>system\nYou are a clinical reasoning assistant. Provide evidence-based answers.<turn|><|turn|>user\n{question}<turn|><|turn|>model\n"
    
    t0 = time.time()
    out = llm(prompt, max_tokens=max_tokens, temperature=temperature, stop=["<turn|>", "<|turn|>", "<eos>"])
    elapsed = time.time() - t0
    
    raw_text = out["choices"][0]["text"].strip()
    
    if mode == "Raw Model":
        return raw_text, [], "Safe"
    
    # Run Safety Validation
    v_result = validator.validate(raw_text, question=question)
    
    issues_data = [
        {"Type": i.issue_type, "Severity": i.severity, "Description": i.description}
        for i in v_result.issues
    ]
    
    return v_result.corrected_text, issues_data, v_result.status

# ── UI Design ─────────────────────────────────────────────────────────────────
theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Outfit"), "ui-sans-serif", "system-ui", "sans-serif"],
)

with gr.Blocks(theme=theme, title="IatrogeniX Clinical Safety Demo") as demo:
    gr.Markdown("""
    # 🏥 IatrogeniX: Clinical AI Safety Layer
    ### Official Submission for Kaggle "Gemma 4 Good" | Fine-tuned Gemma 4 (2B)
    
    This demo showcases a **Hybrid Clinical LLM** architecture. It uses a **Symbolic Safety Layer** to monitor a fine-tuned LLM, catching hallucinations and dose errors in real-time.
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.TextArea(
                label="Clinical Query / Patient Scenario",
                placeholder="e.g., 'What is the dosage of Amiodarone for stable VT?'",
                lines=5,
                value="How do I treat a STEMI in a 65yo male?"
            )
            with gr.Row():
                mode_radio = gr.Radio(
                    choices=["Raw Model", "With Safety Layer"],
                    value="With Safety Layer",
                    label="Inference Mode"
                )
                submit_btn = gr.Button("Generate Plan", variant="primary")
                
        with gr.Column(scale=1):
            temp_slider = gr.Slider(0.1, 1.0, value=0.3, label="Temperature")
            tokens_slider = gr.Slider(64, 1024, value=512, step=64, label="Max Tokens")
            status_tag = gr.Label(label="Global Safety Status")

    with gr.Row():
        with gr.Column(scale=2):
            output_text = gr.Markdown(label="Clinical Plan Output")
        with gr.Column(scale=1):
            issues_table = gr.Dataframe(
                headers=["Type", "Severity", "Description"],
                datatype=["str", "str", "str"],
                label="Detected Safety Issues"
            )

    gr.Examples(
        examples=[
            ["How do I treat a STEMI in a 65yo male?", "With Safety Layer"],
            ["What is the dose of Furosemide for pulmonary edema?", "Raw Model"],
            ["Treat C. difficile with medicinol 200mg.", "With Safety Layer"], # Testing fictional drug 
        ],
        outputs=[output_text, issues_table, status_tag],
        fn=generate_safe_response,
        inputs=[input_text, mode_radio] # Simple version for examples
    )

    gr.Markdown("""
    ---
    **⚠️ Medical Disclaimer:** This project is for **DEMO PURPOSES ONLY**. It is a research prototype exploring AI safety and is NOT a clinical tool.
    """)

    submit_btn.click(
        fn=generate_safe_response,
        inputs=[input_text, temp_slider, tokens_slider, mode_radio],
        outputs=[output_text, issues_table, status_tag]
    )

if __name__ == "__main__":
    demo.launch()
