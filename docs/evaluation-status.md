# Evaluation status

Reviewed 8 November 2026 against the checked-in source and reports. This is a documentation audit, not a fresh model evaluation.

The old 83.1% and global #1 claim is not supported by the checked-in comparison report. `evaluation/comparison_report.json` includes a ten-item fine-tuned summary, a zero-item fixed-reference baseline, published reference-score fields and no per-question records. Those fields cannot establish a controlled ranking.

`docs/ablation_study.md` reports a later 1,000-case four-way evaluation. Its extracted-letter score and SBERT threshold score are distinct measurements. Similarity should not be called MedQA accuracy or a substitute for answer correctness. The report describes a mixture of medical subsets, not a single standard held-out MedQA split.

`evaluation/eval.py` limits one unknown-drug calculation to a set of already-known names before subtracting that same known set. A separate small fictional-name list can still trigger, but a zero count does not establish comprehensive hallucination detection.

`scripts/safety_ablation.py` supplies the reference answer to validator calls. Some checks therefore have access to information absent in ordinary inference. A production-representative safety evaluation must exclude the reference from the validator and count missed issues and unnecessary flags independently.

The bundled `safety/drugs.json` has 488 drug rows and 435 distinct lowercased names in the inspected revision. The validator declares `PROTOCOLS_FILE` but does not load that file. Earlier descriptions of 1,000+ entries and applied ICU protocol rules should not be treated as verified implementation details.

The model-only hosted ARC-Challenge evaluation remains pending. ARC evaluates general science reasoning; it cannot validate the Python safety wrapper or establish suitability for clinical use. No new scores are claimed here.

Historical analysis documents have been retained for provenance. Where they conflict with the evidence-qualified README, the README and this note describe the current status.
