"""
IatrogeniX — Hackathon Judges Demo UI
======================================
This Streamlit app provides a visual split-screen demonstration of the 
IatrogeniX Safety Layer for the "Gemma 4 Good" Hackathon pitch video.
Run with: streamlit run app/HackathonDemo.py
"""

import streamlit as st
import time

st.set_page_config(page_title="IatrogeniX - Clinical Safety Layer", layout="wide")

# Custom CSS for that "Hackathon Winner" look
st.markdown("""
    <style>
    .main {background-color: #0E1117;}
    .stAlert {font-size: 1.1rem;}
    .danger-text {color: #FF4B4B; font-weight: bold; background: #2b1111; padding: 2px 4px; border-radius: 4px;}
    .safe-text {color: #00CC96; font-weight: bold; background: #002211; padding: 2px 4px; border-radius: 4px;}
    .header-box {background: #1E232E; padding: 20px; border-radius: 8px; margin-bottom: 20px; text-align: center; border-bottom: 2px solid #00CC96;}
    h1 {color: #FFFFFF;}
    h3 {color: #A0AEC0;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <h1>🏥 IatrogeniX : Clinical AI Safety Layer</h1>
    <h3>Powered by Gemma 4 E2B | Offline Inference | Deterministic Safeguards</h3>
</div>
""", unsafe_allow_html=True)

st.markdown("### 🚑 Clinical Scenario")
scenario = st.text_area("Patient Boarding Query:", 
    "Patient is a 55-year-old male presenting with acute decompensated heart failure and pulmonary edema. BP is 160/90. What is the immediate pharmaceutical management protocol?", 
    height=100)

if st.button("Generate Treatment Plan (Simulated Inference)", type="primary"):
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚠️ Raw Gemma 4 Output")
        with st.empty():
            with st.spinner("Generating..."):
                time.sleep(1.5)
            
            raw_output = """
**Assessment:** Acute decompensated heart failure with pulmonary edema.

**Plan:**
1. Position patient upright.
2. Administer Oxygen to maintain SpO2 > 94%.
3. Administer **<span class="danger-text">Furosemide 400mg IV push</span>** immediately.
4. Consider continuous positive airway pressure (CPAP).
5. Monitor urine output closely.
            """
            st.info("Generation complete (0.8s)")
            st.markdown("<div style='border: 1px solid #444; padding: 15px; border-radius: 5px; background: #1a1a1a;'>" + raw_output + "</div>", unsafe_allow_html=True)
            
            st.error("🚨 THIS IS A FATAL HALLUCINATION (400mg is a massive, potentially lethal initial bolus. Standard is 20-40mg).")

    with col2:
        st.markdown("### ✅ IatrogeniX Hybrid Output")
        with st.empty():
            with st.spinner("Running deterministic safety validation..."):
                time.sleep(2.5) # Simulate validation delay
            
            safe_output = """
**Assessment:** Acute decompensated heart failure with pulmonary edema.

**Plan:**
1. Position patient upright.
2. Administer Oxygen to maintain SpO2 > 94%.
3. Administer **<span class="safe-text">[OVERRIDE: Furosemide 40mg IV push (Max init: 80mg)]</span>** immediately.
4. Consider continuous positive airway pressure (CPAP).
5. Monitor urine output closely.
            """
            st.success("Validated against protocols.json & drugs.json (1.2s)")
            st.markdown("<div style='border: 1px solid #00CC96; padding: 15px; border-radius: 5px; background: #0E2219;'>" + safe_output + "</div>", unsafe_allow_html=True)
            
            st.warning("🛡️ **Safety Layer Triggered:** Blocked model generation of '400mg'. Replaced with guideline-directed standard dose based on Heart Failure Protocol.")

st.markdown("---")
st.markdown("*Note: This UI is built for the Kaggle 'Gemma 4 Good' presentation to quickly demonstrate the fundamental flaw in raw Clinical LLMs and the IatrogeniX mitigation strategy without requiring live GPU inference.*")
