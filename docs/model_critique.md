# Model Critique: Why High Scores Can Be Decieving

You asked "why my model is bad." While our benchmarking shows IatrogeniX (2.6B) is currently the #1 ranked model in the Edge class, it is critical to understand the **failure modes** that these scores hide. 

Here is the honest technical critique of why this model (like most <10B medical LLMs) is not yet ready for real-world high-stakes clinical care.

## 1. The "Einstellung Effect" (Pattern Matching Over Reasoning)
IatrogeniX performs exceptionally well on Multiple-Choice Questions (MedQA/USMLE) because these exams follow predictable, structural clinical vignettes. 
- **The Risk**: The model has likely "memorized" how to solve USMLE-style problems.
- **The Failure**: In a "messy" real-world case where a patient provides noisy, contradictory, or irrelevant history, the model's logic is **fragile**. It lacks the parameter depth to perform multi-step iterative re-evaluation.

## 2. Lack of Metacognition (Knowing what it doesn't know)
One of the most dangerous traits of small LLMs is **uncalibrated overconfidence**.
- **The Risk**: Larger models (70B+) are better at signaling uncertainty (e.g., "I am not sure, but it could be...").
- **The Failure**: IatrogeniX (2.6B) is prone to providing a definitive, authoritative-sounding diagnosis even when it is factually incorrect. It lacks "metacognition"—the ability to recognize its own knowledge limits.

## 3. Benchmark Contamination (The "Cheating" Problem)
Since Gemma 4 is a recent and highly capable model, its pre-training data is vast.
- **The Risk**: There is a high probability that variations of MedQA and PubMed questions were present in the base model's training set.
- **The Failure**: The high scores may reflect **retrieval** (memory) rather than **reasoning**. If we tested it on a strictly private, "blind" clinical exam from this morning’s rounds, its accuracy would likely drop by 20-30%.

## 4. Symbolic Layer vs. Probabilistic Drift
Our Symbolic Safety Layer (the `SafetyValidator`) is a "hard" fix for a "soft" problem.
- **The Risk**: If a drug dose isn't in our `drugs.json` database, the model is left to its own probabilistic devices.
- **The Failure**: The model can still hallucinate a "plausible but lethal" treatment path for any condition not explicitly covered by the symbolic guards. It doesn't "understand" safety; it just follows patterns.

## 5. Parameter Scale and Multi-Step Inference
Clinical decision-making often requires holding 10+ variables (labs, history, vitals, allergies) in active "working memory" and reasoning through their interactions.
- **The Risk**: Small models (2.6B) have limited "contextual bandwidth" for complex logical chains.
- **The Failure**: It can handle "A leads to B," but it struggles with "A, B, and C together make D impossible, so we must do E."

## Summary of "Badness"
| Factor | Benchmark Result | Clinical Reality |
| :--- | :--- | :--- |
| **Logic** | 84.1% MedQA (Excellent) | Fragile under noise |
| **Safety** | Intercepts known errors | Blind to unknown errors |
| **Trust** | Ranks #1 in class | Dangerous overconfidence |

> [!CAUTION]
> **Conclusion**: IatrogeniX is a powerful tool for **clinical documentation assistance** and **educational retrieval**, but it is "bad" for **autonomous clinical decision-making**. It is a reasoning *mirror*, not a reasoning *brain*.
