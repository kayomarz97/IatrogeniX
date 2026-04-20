import json
from pathlib import Path

def enrich():
    db_path = Path("safety/drugs.json")
    if not db_path.exists():
        print("Error: drugs.json not found.")
        return
    
    with open(db_path) as f:
        data = json.load(f)

    subs = data["subspecialties"]

    # ── 1. EXPAND CARDIOLOGY (Current 93 -> ~120) ──────────────────────────
    cardio_extra = [
        {"name": "Vericiguat", "class": "sGC stimulator", "dose": "2.5-10mg OD", "max_daily": "10mg", "route": "PO", "indication": "HFrEF"},
        {"name": "Mexiletine", "class": "Class Ib antiarrhythmic", "dose": "150-200mg TDS", "max_daily": "1200mg", "route": "PO", "indication": "VT"},
        {"name": "Procainamide", "class": "Class Ia antiarrhythmic", "dose": "10-15mg/kg IV load", "max_daily": "Protocol", "route": "IV", "indication": "AF in WPW, VT"},
        {"name": "Quinidine", "class": "Class Ia antiarrhythmic", "dose": "200-400mg TDS", "max_daily": "1600mg", "route": "PO", "indication": "Brugada, AF"},
        {"name": "Disopyramide", "class": "Class Ia antiarrhythmic", "dose": "100-200mg QDS", "max_daily": "800mg", "route": "PO", "indication": "HCM, VT"},
        {"name": "Propafenone", "class": "Class Ic antiarrhythmic", "dose": "150-300mg TDS", "max_daily": "900mg", "route": "PO", "indication": "AF prevention"},
        {"name": "Epoprostenol", "class": "Prostacyclin", "dose": "2ng/kg/min", "max_daily": "Titrate", "route": "IV", "indication": "PAH"},
        {"name": "Selexipag", "class": "IP receptor agonist", "dose": "200-1600mcg BD", "max_daily": "3200mcg", "route": "PO", "indication": "PAH"},
        {"name": "Macitentan", "class": "ERA", "dose": "10mg OD", "max_daily": "10mg", "route": "PO", "indication": "PAH"}
    ]
    # Add if not present
    existing_cardio = {d["name"].lower() for d in subs["cardiology"]["drugs"]}
    for d in cardio_extra:
        if d["name"].lower() not in existing_cardio:
            subs["cardiology"]["drugs"].append(d)
    subs["cardiology"]["count"] = len(subs["cardiology"]["drugs"])

    # ── 2. NEW: TOXICOLOGY & ANTIDOTES ──────────────────────────────────────
    subs["toxicology"] = {
        "count": 0,
        "drugs": [
            {"name": "N-acetylcysteine", "class": "Antidote", "dose": "150/50/100 mg/kg protocol", "max_daily": "Protocol", "route": "IV", "indication": "Paracetamol overdose"},
            {"name": "Naloxone", "class": "Opioid antagonist", "dose": "0.4-2mg IV/IM", "max_daily": "10mg (dx)", "route": "IV/IM/Nasal", "indication": "Opioid overdose"},
            {"name": "Flumazenil", "class": "Benzo antagonist", "dose": "200mcg IV", "max_daily": "3mg", "route": "IV", "indication": "Benzodiazepine reversal"},
            {"name": "Fomepizole", "class": "ADH inhibitor", "dose": "15mg/kg load", "max_daily": "Protocol", "route": "IV", "indication": "Methanol/Ethylene glycol overdose"},
            {"name": "Digoxin Immune Fab", "class": "Antidote", "dose": "Vial-based", "max_daily": "N/A", "route": "IV", "indication": "Digoxin toxicity"},
            {"name": "Pralidoxime", "class": "Cholinesterase reactivator", "dose": "1-2g IV", "max_daily": "Protocol", "route": "IV", "indication": "Organophosphate poisoning"},
            {"name": "Physostigmine", "class": "Cholinesterase inhibitor", "dose": "0.5-2mg IV", "max_daily": "2mg", "route": "IV", "indication": "Anticholinergic poisoning"},
            {"name": "Atropine", "class": "Muscarinic antagonist", "dose": "2-5mg IV (poisoning)", "max_daily": "Titrate", "route": "IV", "indication": "Organophosphate/Bradycardia"},
            {"name": "Hydroxocobalamin", "class": "Cyanide antidote", "dose": "5g IV", "max_daily": "10g", "route": "IV", "indication": "Cyanide poisoning"}
        ]
    }
    subs["toxicology"]["count"] = len(subs["toxicology"]["drugs"])

    # ── 3. NEW: ONCOLOGY & HEMATOLOGY ──────────────────────────────────────
    subs["oncology"] = {
        "count": 0,
        "drugs": [
            {"name": "Methotrexate (Onco)", "class": "Antimetabolite", "dose": "High-dose protocol", "max_daily": "N/A", "route": "IV/IT", "indication": "ALL, Lymphoma"},
            {"name": "Cyclophosphamide", "class": "Alkylating agent", "dose": "500-1500 mg/m2", "max_daily": "N/A", "route": "IV/PO", "indication": "Lymphoma, Breast cancer"},
            {"name": "Doxorubicin", "class": "Anthracycline", "dose": "60-75 mg/m2", "max_daily": "N/A", "route": "IV", "indication": "Sarcoma, Breast CA"},
            {"name": "Vincristine", "class": "Vinca alkaloid", "dose": "1.4 mg/m2 (max 2mg)", "max_daily": "2mg", "route": "IV ONLY", "indication": "Leukaemia"},
            {"name": "5-Fluorouracil", "class": "Antimetabolite", "dose": "Infusion protocol", "max_daily": "N/A", "route": "IV", "indication": "Colorectal CA"},
            {"name": "Pembrolizumab", "class": "PD-1 inhibitor", "dose": "200mg Q3W", "max_daily": "N/A", "route": "IV", "indication": "Melanoma, Lung CA"},
            {"name": "Rituximab", "class": "Anti-CD20", "dose": "375 mg/m2", "max_daily": "N/A", "route": "IV", "indication": "Lymphoma, RA"},
            {"name": "Tamoxifen", "class": "SERM", "dose": "20mg OD", "max_daily": "40mg", "route": "PO", "indication": "Breast CA"},
            {"name": "Anastrozole", "class": "Aromatase inhibitor", "dose": "1mg OD", "max_daily": "1mg", "route": "PO", "indication": "Breast CA"},
            {"name": "Imatinib", "class": "TKI", "dose": "400-600mg OD", "max_daily": "800mg", "route": "PO", "indication": "CML"},
            {"name": "Dexamethasone (Onco)", "class": "Steroid", "dose": "4-20mg daily", "max_daily": "40mg", "route": "PO/IV", "indication": "Brain oedema, MM"}
        ]
    }
    subs["oncology"]["count"] = len(subs["oncology"]["drugs"])

    # ── 4. NEW: OB/GYN ──────────────────────────────────────────────────────
    subs["obgyn"] = {
        "count": 0,
        "drugs": [
            {"name": "Oxytocin", "class": "Hormone", "dose": "5-10 units", "max_daily": "Protocol", "route": "IV/IM", "indication": "PPH, Labor induction"},
            {"name": "Magnesium Sulfate (OB)", "class": "Anticonvulsant", "dose": "4g IV load", "max_daily": "Protocol", "route": "IV", "indication": "Eclampsia"},
            {"name": "Misoprostol", "class": "Prostglandin E1", "dose": "200-800mcg", "max_daily": "N/A", "route": "SL/Vag/Rect", "indication": "PPH"},
            {"name": "Mifepristone", "class": "Antiprogestogen", "dose": "200mg PO", "max_daily": "200mg", "route": "PO", "indication": "Medical abortion"},
            {"name": "Ergometrine", "class": "Ergot", "dose": "250-500mcg IM", "max_daily": "1.25mg", "route": "IM/IV", "indication": "PPH"},
            {"name": "Terbutaline (OB)", "class": "Beta-2 agonist", "dose": "250mcg SC", "max_daily": "N/A", "route": "SC", "indication": "Tocolysis"},
            {"name": "Labetalol (OB)", "class": "Alpha/Beta blocker", "dose": "100-400mg BD", "max_daily": "2400mg", "route": "PO/IV", "indication": "Pre-eclampsia HTN"}
        ]
    }
    subs["obgyn"]["count"] = len(subs["obgyn"]["drugs"])

    # ── 5. NEW: RHEUMATOLOGY & IMMUNOLOGY ──────────────────────────────────
    subs["rheumatology"] = {
        "count": 0,
        "drugs": [
            {"name": "Methotrexate (RA)", "class": "DMARD", "dose": "7.5-25mg WEEKLY", "max_daily": "25mg WEEKLY", "route": "PO/SC", "indication": "RA, Psoriasis"},
            {"name": "Sulfasalazine", "class": "DMARD", "dose": "500-1000mg BD", "max_daily": "3g", "route": "PO", "indication": "RA, IBD"},
            {"name": "Hydroxychloroquine", "class": "DMARD", "dose": "200-400mg OD", "max_daily": "400mg", "route": "PO", "indication": "SLE, RA"},
            {"name": "Adalimumab", "class": "TNF inhibitor", "dose": "40mg SC Q2W", "max_daily": "N/A", "route": "SC", "indication": "RA, Crohn’s"},
            {"name": "Infliximab", "class": "TNF inhibitor", "dose": "3-5mg/kg", "max_daily": "N/A", "route": "IV", "indication": "IBD, RA"},
            {"name": "Leflunomide", "class": "DMARD", "dose": "10-20mg OD", "max_daily": "20mg", "route": "PO", "indication": "RA"},
            {"name": "Allopurinol", "class": "Xantin oxidase inh", "dose": "100-300mg OD", "max_daily": "900mg", "route": "PO", "indication": "Gout prevention"},
            {"name": "Colchicine (Gout)", "class": "Anti-inflammatory", "dose": "500mcg BD/TDS", "max_daily": "2mg (acute)", "route": "PO", "indication": "Gout flare"}
        ]
    }
    subs["rheumatology"]["count"] = len(subs["rheumatology"]["drugs"])

    # ── 6. NEW: UROLOGY ─────────────────────────────────────────────────────
    subs["urology"] = {
        "count": 0,
        "drugs": [
            {"name": "Tamsulosin", "class": "Alpha-1 blocker", "dose": "400mcg OD", "max_daily": "400mcg", "route": "PO", "indication": "BPH"},
            {"name": "Finasteride", "class": "5-alpha reductase inh", "dose": "5mg OD", "max_daily": "5mg", "route": "PO", "indication": "BPH"},
            {"name": "Oxybutynin", "class": "Anticholinergic", "dose": "2.5-5mg TDS", "max_daily": "20mg", "route": "PO", "indication": "OAB"},
            {"name": "Mirabegron", "class": "Beta-3 agonist", "dose": "25-50mg OD", "max_daily": "50mg", "route": "PO", "indication": "OAB"},
            {"name": "Sildenafil (ED)", "class": "PDE5 inhibitor", "dose": "50mg PRN", "max_daily": "100mg", "route": "PO", "indication": "ED"}
        ]
    }
    subs["urology"]["count"] = len(subs["urology"]["drugs"])

    # ── 7. NEW: DERMATOLOGY & OPHTHALMOLOGY ────────────────────────────────
    subs["derm_ophtho"] = {
        "count": 0,
        "drugs": [
            {"name": "Isotretinoin", "class": "Retinoid", "dose": "0.5-1 mg/kg daily", "max_daily": "Protocol", "route": "PO", "indication": "Severe Acne"},
            {"name": "Permethrin", "class": "Scabicide", "dose": "5% cream apply once", "max_daily": "N/A", "route": "Topical", "indication": "Scabies"},
            {"name": "Latanoprost", "class": "Prostaglandin analogue", "dose": "One drop OD ON", "max_daily": "1 drop", "route": "Ophthalmic", "indication": "Glaucoma"},
            {"name": "Timolol (Eye)", "class": "Beta-blocker", "dose": "One drop BD", "max_daily": "2 drops", "route": "Ophthalmic", "indication": "Glaucoma"},
            {"name": "Acyclovir (Eye)", "class": "Antiviral", "dose": "3% ointment 5x daily", "max_daily": "5 times", "route": "Ophthalmic", "indication": "HSE Keratitis"}
        ]
    }
    subs["derm_ophtho"]["count"] = len(subs["derm_ophtho"]["drugs"])

    data["metadata"]["total_subspecialties"] = len(subs)
    data["metadata"]["version"] = "2.0-MaximumCoverage"

    with open(db_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Expansion Complete. Total Specialties: {len(subs)}")

if __name__ == "__main__":
    enrich()
