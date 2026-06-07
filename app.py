from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

from predictor_core import (
    METADATA,
    THRESHOLD,
    calculate_suv_ratio,
    predict_single,
    predict_batch,
)

st.set_page_config(
    page_title="PET/CT pCR Prediction Calculator",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #fbfcfe; }
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1180px; }
    .hero {
        background: linear-gradient(135deg, #17324D 0%, #235f7a 52%, #2f855a 100%);
        padding: 1.45rem 1.6rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(23,50,77,0.14);
    }
    .hero h1 { margin: 0; font-size: 2.0rem; font-weight: 750; }
    .hero p { margin: 0.45rem 0 0 0; color: rgba(255,255,255,0.88); font-size: 1.02rem; }
    .card {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 4px 18px rgba(15,23,42,0.045);
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #dbe4ee;
        border-radius: 15px;
        padding: 1.0rem;
        height: 100%;
    }
    .risk-positive {
        color: #166534;
        font-weight: 750;
        font-size: 1.15rem;
    }
    .risk-negative {
        color: #9a3412;
        font-weight: 750;
        font-size: 1.15rem;
    }
    .small-note { color: #64748b; font-size: 0.88rem; line-height: 1.45; }
    .footer-note {
        border-left: 4px solid #94a3b8;
        padding-left: 0.85rem;
        color: #475569;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>PET/CT-based pCR Prediction Calculator</h1>
      <p>Random Forest model for predicting pathological complete response after neoadjuvant therapy in NSCLC.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Model information")
    st.write(f"**Final model:** {METADATA['model_name']}")
    st.write(f"**Outcome:** {METADATA['outcome']}")
    st.write(f"**Decision threshold:** {THRESHOLD:.3f}")
    st.divider()
    perf = METADATA["performance"]
    st.caption("Reported performance")
    st.write(f"Training AUC: **{perf['training_auc']:.3f}**")
    st.write(f"Bootstrap-OOB AUC: **{perf['bootstrap_oob_auc']:.3f}**")
    st.write(f"Validation AUC: **{perf['validation_auc']:.3f}**")
    st.write(f"Validation Brier score: **{perf['validation_brier_score']:.3f}**")
    st.divider()
    st.warning("Research use only. This tool is not intended to replace clinical judgment.")

single_tab, batch_tab, info_tab = st.tabs(["Single-patient prediction", "Batch prediction", "Model notes"])

with single_tab:
    st.subheader("Enter patient-level predictors")
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### PET/CT metabolic parameters")
        c1, c2 = st.columns(2)
        with c1:
            suvmax1 = st.number_input(
                "Pre-treatment SUVmax (SUVmax1)", min_value=0.01, max_value=80.0,
                value=16.10, step=0.10, format="%.2f",
                help="Baseline SUVmax before neoadjuvant therapy."
            )
        with c2:
            suvmax2 = st.number_input(
                "Post-treatment SUVmax (SUVmax2)", min_value=0.0, max_value=80.0,
                value=4.80, step=0.10, format="%.2f",
                help="SUVmax after neoadjuvant therapy."
            )
        auto_ratio = calculate_suv_ratio(suvmax1, suvmax2)
        use_auto_ratio = st.checkbox(
            "Automatically calculate SUV ratio = (SUVmax1 - SUVmax2) / SUVmax1",
            value=True,
        )
        if use_auto_ratio:
            suv_ratio = auto_ratio
            st.info(f"Calculated SUV ratio: **{suv_ratio:.3f}**")
        else:
            suv_ratio = st.number_input(
                "SUV ratio", min_value=-3.0, max_value=1.0,
                value=float(auto_ratio), step=0.01, format="%.3f"
            )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Clinical and response variables")
        smoking_label = st.radio("Smoking history", ["No", "Yes"], horizontal=True)
        smoking = 1 if smoking_label == "Yes" else 0
        recist_label = st.selectbox(
            "RECIST response",
            ["Responder (CR/PR)", "Non-responder (SD/PD)"],
            help="Responder was defined as CR or PR; non-responder as SD or PD."
        )
        recist_response = "Responder" if recist_label.startswith("Responder") else "Non-responder"
        hist_label = st.selectbox(
            "Histology",
            ["LUAD - lung adenocarcinoma", "LUSC - lung squamous cell carcinoma"]
        )
        histological = "LUAD" if hist_label.startswith("LUAD") else "LUSC"
        cn_label = st.selectbox("Clinical nodal status", ["cN0-1", "cN2"])
        cn_status = cn_label
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Prediction result")
        if st.button("Calculate pCR probability", type="primary", use_container_width=True):
            result = predict_single(
                suvmax1=suvmax1,
                suvmax2=suvmax2,
                suv_ratio=suv_ratio,
                smoking=smoking,
                recist_response=recist_response,
                histological=histological,
                cn_status=cn_status,
            )
            prob = result["pcr_probability"]
            pct = result["pcr_probability_percent"]
            pred = result["classification"]
            band = result["probability_band"]

            st.metric("Predicted pCR probability", f"{pct:.1f}%")
            st.progress(min(max(prob, 0.0), 1.0))
            css_class = "risk-positive" if prob >= THRESHOLD else "risk-negative"
            st.markdown(f'<div class="{css_class}">{pred}</div>', unsafe_allow_html=True)
            st.write(f"Probability band: **{band}**")
            st.write(f"Decision threshold: **{THRESHOLD:.3f}**")

            st.divider()
            st.markdown("##### Input sent to model")
            input_df = pd.DataFrame([result["input"]])
            st.dataframe(input_df, use_container_width=True, hide_index=True)
        else:
            st.markdown(
                """
                <div class="metric-card">
                    <b>Waiting for input</b><br>
                    <span class="small-note">Fill in the predictors and click the calculation button to obtain the predicted pCR probability.</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

with batch_tab:
    st.subheader("Batch prediction from CSV")
    st.write("Upload a CSV file containing the seven model predictors. If `SUV_ratio` is absent, it will be calculated automatically from `SUVmax1` and `SUVmax2`.")

    template = pd.DataFrame({
        "SUVmax1": [16.1, 20.3],
        "SUVmax2": [4.8, 3.5],
        "SUV_ratio": [0.702, 0.828],
        "Smoking": [1, 0],
        "RECIST_response": ["Responder", "Non-responder"],
        "Histological": ["LUAD", "LUSC"],
        "cN": ["cN2", "cN0-1"],
    })
    st.download_button(
        "Download CSV template",
        data=template.to_csv(index=False).encode("utf-8-sig"),
        file_name="pcr_prediction_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            result_df = predict_batch(df)
            st.success(f"Prediction completed for {len(result_df)} patients.")
            st.dataframe(result_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download prediction results",
                data=result_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="pcr_prediction_results.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with info_tab:
    st.subheader("How this calculator works")
    st.markdown(
        f"""
        This web calculator uses the final Random Forest model developed from seven selected predictors:
        **SUVmax1, SUVmax2, SUV ratio, RECIST response, Smoking, Histology, and cN status**.

        - Continuous variables are standardized inside the saved sklearn pipeline.
        - Categorical variables are one-hot encoded inside the saved sklearn pipeline.
        - The predicted value is the estimated probability of pathological complete response.
        - The default decision threshold is **{THRESHOLD:.3f}**, derived from aggregated bootstrap-OOB analysis.

        **Important limitation:** this model was developed from a single-center retrospective dataset and should be externally validated before routine clinical use.
        """
    )
    st.markdown(
        """
        <div class="footer-note">
        Suggested citation text for manuscript: An online prediction calculator was implemented based on the final Random Forest pipeline. The calculator accepts the seven selected predictors and returns an individualized pCR probability using the same preprocessing and model parameters as the final analytical model.
        </div>
        """,
        unsafe_allow_html=True,
    )
