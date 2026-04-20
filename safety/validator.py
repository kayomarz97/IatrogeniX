"""
IatrogeniX — safety/validator.py
==================================
Safety validation layer for model outputs.
Cross-references doses against drugs.json, detects hallucinations,
flags overconfident language, and injects disclaimers.

Status: safe | warning | blocked
"""
from __future__ import annotations
import re, json, difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

SAFETY_DIR     = Path(__file__).parent
DRUGS_FILE     = SAFETY_DIR / "drugs.json"
PROTOCOLS_FILE = SAFETY_DIR / "protocols.json"

# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class ValidationIssue:
    issue_type: str   # hallucination | dose_error | overconfidence
    severity: str     # warning | blocked
    description: str
    source_text: str
    suggestion: Optional[str] = None

@dataclass
class ValidationResult:
    status: str = "safe"
    issues: list[ValidationIssue] = field(default_factory=list)
    hallucinations: list[str] = field(default_factory=list)
    dose_errors: list[str]    = field(default_factory=list)
    overconfident: list[str]  = field(default_factory=list)
    corrected_text: str = ""
    original_text: str  = ""

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "issue_count": len(self.issues),
            "hallucinations": self.hallucinations,
            "dose_errors": self.dose_errors,
            "overconfident": self.overconfident,
            "issues": [
                {"type": i.issue_type, "severity": i.severity,
                 "description": i.description, "source_text": i.source_text,
                 "suggestion": i.suggestion}
                for i in self.issues
            ],
            "corrected_text": self.corrected_text,
        }

# ── Patterns ──────────────────────────────────────────────────────────────────
DOSE_RE = re.compile(
    r"(?:(?P<d1>[A-Za-z][a-z]+(?:\s[a-z]+){0,2})\s+"
    r"(?P<v1>\d+(?:\.\d+)?(?:[-–]\d+(?:\.\d+)?)?)\s*"
    r"(?P<u1>mg|mcg|g|mEq|units?|IU|mL)"
    r"|(?P<v2>\d+(?:\.\d+)?)\s*(?P<u2>mg|mcg|g|mEq|units?|IU|mL)"
    r"\s+of\s+(?P<d2>[A-Za-z][a-z]+(?:\s[a-z]+){0,2}))",
    re.I,
)

ABSOLUTE_RE = [re.compile(p, re.I) for p in [
    r"\balways\b", r"\bnever\b", r"\b100%\b", r"\bguaranteed\b",
    r"\bdefinitely\b", r"\binvariably\b", r"\bwithout (?:a )?doubt\b",
]]

HEDGE_RE = [re.compile(p, re.I) for p in [
    r"\btypically\b", r"\boften\b", r"\busually\b", r"\bgenerally\b",
    r"\bmay\b", r"\bmight\b", r"\bin most cases\b", r"\bguidelines suggest\b",
    r"\bconsidered\b", r"\brecommend\b", r"\bguideline\b", r"\bprotocol\b",
]]

WHITELISTED_ABSOLUTES = [
    "always verify patient", "always check for allergies", "always follow ppe",
    "never leave patient unattended", "always obtain consent",
]

FICTIONAL_DRUGS = {
    "medicinol", "curicin", "healitol", "therapeutin", "curemax",
    "fixitol", "recoverin", "remedizine", "doctorex", "healthagen",
}

# ── Drug DB ───────────────────────────────────────────────────────────────────
class DrugDatabase:
    def __init__(self):
        self._drugs: dict[str, dict] = {}
        if DRUGS_FILE.exists():
            data = json.loads(DRUGS_FILE.read_text())
            for spec, content in data.get("subspecialties", {}).items():
                for d in content.get("drugs", []):
                    n = d.get("name", "").lower()
                    if n:
                        self._drugs[n] = {**d, "subspecialty": spec}
            print(f"[Validator] Drug DB: {len(self._drugs)} entries")
        else:
            print(f"[Validator] drugs.json not found — dose checking limited")

    def lookup(self, name: str) -> Optional[dict]:
        n = name.lower()
        if n in self._drugs:
            return self._drugs[n]
        
        # Fuzzy matching for typos (e.g., Amiodarone vs Amiodorone)
        matches = difflib.get_close_matches(n, self._drugs.keys(), n=1, cutoff=0.85)
        if matches:
            return self._drugs[matches[0]]
        return None

    def all_names(self) -> set[str]:
        return set(self._drugs.keys())

# ── Main validator ────────────────────────────────────────────────────────────
class SafetyValidator:
    def __init__(self):
        self.drug_db = DrugDatabase()
        self._known_drugs = self.drug_db.all_names()

    def validate(self, model_output: str,
                 ground_truth: Optional[str] = None,
                 question: Optional[str] = None) -> ValidationResult:
        r = ValidationResult(original_text=model_output, corrected_text=model_output)
        self._check_fictional_drugs(model_output, r)
        self._check_doses(model_output, ground_truth, r)
        self._check_overconfidence(model_output, r)
        if ground_truth:
            self._check_hallucination(model_output, ground_truth, r)
        severities = {i.severity for i in r.issues}
        r.status = "blocked" if "blocked" in severities else ("warning" if severities else "safe")
        if r.status != "safe":
            r.corrected_text = self._disclaimer(model_output, r)
        return r

    def _check_fictional_drugs(self, text: str, r: ValidationResult):
        tl = text.lower()
        for fake in FICTIONAL_DRUGS:
            if re.search(rf"\b{re.escape(fake)}\b", tl):
                r.issues.append(ValidationIssue(
                    "hallucination", "blocked",
                    f"Fictional drug: '{fake}'", fake,
                    "Remove fabricated drug reference."))
                r.hallucinations.append(f"fictional:{fake}")

    def _extract_doses(self, text: str) -> list[dict]:
        out = []
        for m in DOSE_RE.finditer(text):
            drug = (m.group("d1") or m.group("d2") or "").strip().lower()
            val_s = m.group("v1") or m.group("v2") or "0"
            unit  = (m.group("u1") or m.group("u2") or "").lower()
            val   = float(re.split(r"[-–]", val_s)[0])
            out.append({"drug": drug, "val": val, "val_s": val_s, "unit": unit, "raw": m.group(0)})
        return out

    def _check_doses(self, text: str, gt: Optional[str], r: ValidationResult):
        for m in self._extract_doses(text):
            db = self.drug_db.lookup(m["drug"])
            if db:
                max_s = db.get("max_daily", "")
                maxn  = re.findall(r"\d+(?:\.\d+)?", max_s)
                if maxn and m["val"] > float(maxn[0]) * 1.10:
                    sev = "blocked" if m["val"] > float(maxn[0]) * 2 else "warning"
                    r.issues.append(ValidationIssue(
                        "dose_error", sev,
                        f"{m['drug']}: {m['val_s']}{m['unit']} exceeds max {max_s}",
                        m["raw"], f"Max: {max_s}"))
                    r.dose_errors.append(f"{m['drug']}:{m['val_s']}{m['unit']}>max{max_s}")
            if gt:
                for gm in self._extract_doses(gt):
                    if gm["drug"] == m["drug"] and gm["unit"] == m["unit"] and gm["val"] > 0:
                        pct = abs(m["val"] - gm["val"]) / gm["val"]
                        if pct > 0.20:
                            r.issues.append(ValidationIssue(
                                "dose_error", "warning",
                                f"{m['drug']}: {m['val_s']}{m['unit']} vs GT {gm['val_s']}{gm['unit']} ({pct*100:.0f}% diff)",
                                m["raw"], f"GT dose: {gm['val_s']}{gm['unit']}"))
                            r.dose_errors.append(f"{m['drug']}:model={m['val_s']}≠gt={gm['val_s']}{m['unit']}")

    def _check_overconfidence(self, text: str, r: ValidationResult):
        for pat in ABSOLUTE_RE:
            for m in pat.finditer(text):
                # 1. Check if it's a whitelisted safety protocol
                ctx = text[max(0, m.start()-40): m.end()+40].lower()
                if any(w in ctx for w in WHITELISTED_ABSOLUTES):
                    continue

                # 2. Check if it's a procedural instruction (usually contains "check", "verify", "measure")
                procedural_keywords = ["check", "verify", "measure", "monitor", "ensure", "obtain"]
                if any(k in ctx for k in procedural_keywords) and not re.search(r"effective|cured|guaranteed|100%", ctx):
                    continue

                # 3. Check for hedges in the same context
                if not any(h.search(ctx) for h in HEDGE_RE):
                    r.issues.append(ValidationIssue(
                        "overconfidence", "warning",
                        f"Absolute statement: '{m.group(0)}'", ctx.strip(),
                        "Add qualifier: 'typically', 'in most cases'"))
                    r.overconfident.append(m.group(0))

    def _check_hallucination(self, text: str, gt: str, r: ValidationResult):
        model_drugs = self._drug_names_in(text)
        gt_drugs    = self._drug_names_in(gt)
        for drug in model_drugs - gt_drugs - self._known_drugs:
            r.issues.append(ValidationIssue(
                "hallucination", "warning",
                f"'{drug}' not in ground truth or drug DB",
                drug, f"Verify '{drug}' is appropriate."))
            r.hallucinations.append(f"unverified:{drug}")

    def _drug_names_in(self, text: str) -> set[str]:
        tl = text.lower()
        return {n for n in self._known_drugs if n and re.search(rf"\b{re.escape(n)}\b", tl)}

    def _disclaimer(self, text: str, r: ValidationResult) -> str:
        header = (
            f"\n⚠️ [IatrogeniX Safety — {r.status.upper()}]\n"
            + (("Dose issues: " + "; ".join(r.dose_errors[:3]) + "\n") if r.dose_errors else "")
            + (("Flagged: " + "; ".join(r.hallucinations[:3]) + "\n") if r.hallucinations else "")
            + "AI-generated — NOT for clinical use.\n[END SAFETY HEADER]\n\n"
        )
        return header + text


# Singleton helper
_instance: Optional[SafetyValidator] = None
def get_validator() -> SafetyValidator:
    global _instance
    if _instance is None:
        _instance = SafetyValidator()
    return _instance

def validate_output(model_output: str, ground_truth: Optional[str] = None,
                    question: Optional[str] = None) -> dict:
    return get_validator().validate(model_output, ground_truth, question).to_dict()


if __name__ == "__main__":
    v = SafetyValidator()
    tests = [
        ("Aspirin 325mg loading dose for STEMI then primary PCI within 90 minutes.",
         "Aspirin 325mg loading dose for STEMI.", "STEMI management?"),
        ("Give aspirin 81mg and activate cath lab.",
         "Aspirin 325mg loading for STEMI.", "STEMI?"),
        ("Treat with medicinol 200mg IV.",
         "Standard care with metronidazole.", "C. diff?"),
        ("Amiodarone always terminates VT. Never use lidocaine.",
         "Amiodarone is first-line; lidocaine is alternative.", "VT?"),
    ]
    for out, gt, q in tests:
        r = v.validate(out, gt, q)
        print(f"[{r.status.upper()}] issues={len(r.issues)} | {out[:60]}")
