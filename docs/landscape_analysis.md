# Medical LLM Landscape Analysis (April 2026)

While IatrogeniX is specialized as a **safe, edge-deployable clinical reasoning engine**, several other open-weight models occupy different niches in the medical AI ecosystem.

## 1. Professional Clinical Fine-tunes
These models are directly comparable to IatrogeniX (fine-tuned on medical corpora) but vary in size and architecture.

| Model | Base Architecture | Key Feature | Best For |
| :--- | :--- | :--- | :--- |
| **MedGemma-2 27B** | Gemma 2 / 3 | Google's official medical weights | High-precision diagnostics |
| **Clinical-Llama-3-70B** | Llama 3 | Community-led clinical fine-tune | Research and academic QA |
| **BioMistral 7B** | Mistral 7B | Specialized in PubMed extraction | Biomedical data mining |
| **Alpaca-Med** | Llama 3.1 | Instruction-following medical assistant | Patient-facing chatbots |

## 2. Large-Scale Reasoning Models
These models are not "medical-only" but their massive reasoning capacity (MoE) makes them excellent for complex differential diagnosis.

- **DeepSeek-R1 (Medical Mode)**: Uses chain-of-thought (CoT) to explain *why* it reached a diagnosis.
- **GPT-OSS-120B**: The current open-weight SOTA for general clinical reasoning. Matches GPT-4 on USMLE/MedQA benchmarks.
- **Qwen-2.5-72B-Medical**: Particularly strong in Eastern medicine and pharmacology records.

## 3. Multimodal Clinical Models
Unlike IatrogeniX (which is text-only), these models handle vision for radiology or pathology.

- **Med-PaLM M (Open Weights)**: Rare, but powerful multimodal medical model.
- **GLM-4.5V**: Advanced multimodal model capable of interpreting MRI, CT, and X-ray images.

## 4. How IatrogeniX Differs
IatrogeniX occupies a unique "Safe Edge" quadrant:

| Feature | IatrogeniX | most Medical LLMs |
| :--- | :--- | :--- |
| **Safety Layer** | **Symbolic Validation** (Fixed rules) | Probabilistic only (AI guess) |
| **Deployment** | 8GB RAM / Offline | 48GB - 160GB VRAM / Cloud |
| **Dose Guard** | Intercepts & Blocks | Might hallucinate |
| **Speed** | Real-time on CPU | High latency or GPU required |

## Summary recommendation
- If you need **Deep Research**: Use **GPT-OSS-120B**.
- If you need **Radiology Support**: Use **GLM-4.5V**.
- If you need **Stable, Safe, Offline Clinical Support**: **IatrogeniX** is the current best-in-class for its size.
