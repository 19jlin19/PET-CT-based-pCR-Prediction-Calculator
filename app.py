from __future__ import annotations

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
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #17324d;
        --blue: #1f5d7a;
        --teal: #198f91;
        --ink: #172033;
        --muted: #64748b;
        --line: #dbe4ee;
        --soft: #f5f8fb;
        --green: #157347;
        --orange: #b45309;
    }
    .stApp { background: #f7f9fc; }
    .block-container { max-width: 1220px; padding-top: 1.1rem; padding-bottom: 2.4rem; }
    [data-testid="stSidebar"] { background: #f0f5f9; }
    .hero {
        background: linear-gradient(118deg, #17324d 0%, #1f5d7a 58%, #198f91 100%);
        border-radius: 20px;
        padding: 1.65rem 1.8rem 1.55rem 1.8rem;
        color: white;
        margin-bottom: 1.1rem;
        box-shadow: 0 10px 30px rgba(23,50,77,.16);
    }
    .hero-kicker { font-size: .78rem; letter-spacing: .13em; text-transform: uppercase; opacity: .80; font-weight: 700; }
    .hero h1 { margin: .28rem 0 .35rem 0; font-size: 2.15rem; line-height: 1.15; font-weight: 800; }
    .hero p { margin: 0; max-width: 890px; color: rgba(255,255,255,.88); font-size: 1.02rem; line-height: 1.52; }
    .hero-badge { display: inline-block; margin-top: .9rem; padding: .32rem .68rem; border: 1px solid rgba(255,255,255,.35); border-radius: 999px; font-size: .78rem; color: rgba(255,255,255,.92); }
    .section-label { color: var(--navy); font-size: 1.08rem; font-weight: 780; margin: .25rem 0 .25rem 0; }
    .subtle { color: var(--muted); font-size: .9rem; line-height: 1.48; }
    .panel { background: #fff; border: 1px solid var(--line); border-radius: 17px; padding: 1.05rem 1.15rem; box-shadow: 0 5px 18px rgba(15,23,42,.045); height: 100%; }
    .panel-title { color: var(--navy); font-size: 1.04rem; font-weight: 780; margin-bottom: .65rem; }
    .result-empty { background: linear-gradient(180deg,#f8fbfd,#f2f7fa); border: 1px dashed #b9cbd8; border-radius: 16px; padding: 2.2rem 1.2rem; text-align: center; color: var(--muted); }
    .result-prob { color: var(--navy); font-size: 3.1rem; line-height: 1; font-weight: 850; letter-spacing: -.04em; }
    .result-label { color: var(--muted); font-size: .9rem; margin-top: .3rem; }
    .status-good, .status-low { display: inline-block; margin-top: .85rem; padding: .42rem .75rem; border-radius: 999px; font-weight: 760; font-size: .92rem; }
    .status-good { color: #166534; background: #dcfce7; }
    .status-low { color: #9a3412; background: #ffedd5; }
    .method-note { border-left: 4px solid var(--teal); padding: .65rem .85rem; color: #475569; background: #f4fbfb; border-radius: 0 10px 10px 0; font-size: .88rem; line-height: 1.5; }
    .footer-note { border-top: 1px solid var(--line); margin-top: 1.2rem; padding-top: .85rem; color: var(--muted); font-size: .82rem; line-height: 1.45; }
    div[data-testid="stMetric"] { background: #fff; border: 1px solid var(--line); border-radius: 12px; padding: .65rem .75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

perf = METADATA["performance"]

st.markdown(
    f"""
    <div class="hero">
      <div class="hero-kicker">Research prototype · NSCLC · PET/CT</div>
      <h1>PET/CT-based pCR Prediction Calculator</h1>
      <p>Individualized estimation of pathological complete response after neoadjuvant chemoimmunotherapy using a seven-predictor Random Forest model.</p>
      <span class="hero-badge">Primary internal validation: training-only repeated nested cross-validation</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Model summary")
    st.write(f"**Model:** {METADATA['model_name']}")
    st.write(f"**Outcome:** {METADATA['outcome']}")
    st.write(f"**Predictors:** {len(METADATA['features'])}")
    st.write(f"**Decision threshold:** {THRESHOLD:.3f}")
    st.divider()
    st.markdown("### Performance overview")
    st.metric("Nested-CV AUC", f"{perf['nested_cv_auc']:.3f}")
    st.caption(f"95% CI {perf['nested_cv_auc_ci95'][0]:.3f}–{perf['nested_cv_auc_ci95'][1]:.3f}")
    st.metric("Validation AUC", f"{perf['validation_auc']:.3f}")
    st.caption(f"95% CI {perf['validation_auc_ci95'][0]:.3f}–{perf['validation_auc_ci95'][1]:.3f}")
    st.metric("Validation Brier score", f"{perf['validation_brier_score']:.3f}")
    st.divider()
    st.caption("Bootstrap-OOB was retained as a supplementary robustness analysis and was not used as the primary model-selection criterion.")
    st.warning("Research use only. This calculator does not replace pathological assessment or clinical judgment.")

single_tab, batch_tab, method_tab = st.tabs(["Single-patient prediction", "Batch prediction", "Model information"])

with single_tab:
    st.markdown('<div class="section-label">Enter patient-level predictors</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">The SUV reduction ratio is calculated automatically from the paired PET/CT measurements unless manual entry is selected.</div>', unsafe_allow_html=True)
    st.write("")
    left, right = st.columns([1.08, .92], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">① Longitudinal PET/CT metabolic parameters</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            suvmax1 = st.number_input("Pre-treatment SUVmax (SUVmax1)", min_value=0.01, max_value=80.0, value=16.10, step=0.10, format="%.2f", help="SUVmax before neoadjuvant treatment.")
        with c2:
            suvmax2 = st.number_input("Post-treatment SUVmax (SUVmax2)", min_value=0.0, max_value=80.0, value=4.80, step=0.10, format="%.2f", help="SUVmax after neoadjuvant treatment.")
        st.latex(r"SUV\ reduction\ ratio = \frac{SUV_{max1}-SUV_{max2}}{SUV_{max1}}")
        auto_ratio = calculate_suv_ratio(suvmax1, suvmax2)
        use_auto_ratio = st.checkbox("Automatically calculate SUV reduction ratio", value=True)
        if use_auto_ratio:
            suv_ratio = auto_ratio
            st.info(f"Calculated SUV reduction ratio: **{suv_ratio:.3f}**")
        else:
            suv_ratio = st.number_input("SUV reduction ratio", min_value=-3.0, max_value=1.0, value=float(auto_ratio), step=0.01, format="%.3f")
        st.markdown('</div>', unsafe_allow_html=True)

        st.write("")
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">② Clinical and radiologic response variables</div>', unsafe_allow_html=True)
        smoking_label = st.radio("Smoking history", ["No", "Yes"], horizontal=True)
        smoking = 1 if smoking_label == "Yes" else 0
        recist_label = st.selectbox("RECIST response", ["Responder (CR/PR)", "Non-responder (SD/PD)"], help="Responder = CR/PR; non-responder = SD/PD.")
        recist_response = "Responder" if recist_label.startswith("Responder") else "Non-responder"
        hist_label = st.selectbox("Histological subtype", ["LUAD - lung adenocarcinoma", "LUSC - lung squamous cell carcinoma"])
        histological = "LUAD" if hist_label.startswith("LUAD") else "LUSC"
        cn_status = st.selectbox("Clinical nodal status", ["cN0-1", "cN2"])
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">③ Individualized prediction</div>', unsafe_allow_html=True)
        if st.button("Calculate pCR probability", type="primary", width="stretch"):
            try:
                result = predict_single(suvmax1=suvmax1, suvmax2=suvmax2, suv_ratio=suv_ratio, smoking=smoking, recist_response=recist_response, histological=histological, cn_status=cn_status)
                prob = result["pcr_probability"]
                pct = result["pcr_probability_percent"]
                pred = result["classification"]
                band = result["probability_band"]
                st.markdown(f'<div class="result-prob">{pct:.1f}%</div><div class="result-label">Predicted probability of pathological complete response</div>', unsafe_allow_html=True)
                st.progress(min(max(prob, 0.0), 1.0))
                status_class = "status-good" if prob >= THRESHOLD else "status-low"
                st.markdown(f'<span class="{status_class}">{pred}</span>', unsafe_allow_html=True)
                st.write(f"Descriptive probability band: **{band}**")
                st.caption(f"Decision threshold (Youden index from aggregated outer-fold out-of-fold predictions): {THRESHOLD:.3f}")
                st.divider()
                st.markdown('<div class="method-note">This probability is an individualized model output. It should complement, not replace, postoperative pathological assessment and multidisciplinary clinical judgment.</div>', unsafe_allow_html=True)
                st.write("")
                st.markdown("**Inputs used by the model**")
                st.dataframe(pd.DataFrame([result["input"]]), width="stretch", hide_index=True)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
        else:
            st.markdown('<div class="result-empty"><b>Ready for calculation</b><br><span class="subtle">Enter the seven predictors and select Calculate pCR probability.</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with batch_tab:
    st.markdown('<div class="section-label">Batch prediction from CSV</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Upload a CSV containing the seven model predictors. If SUV_ratio is omitted, it will be calculated automatically from SUVmax1 and SUVmax2.</div>', unsafe_allow_html=True)
    template = pd.DataFrame({"SUVmax1": [16.1, 20.3], "SUVmax2": [4.8, 3.5], "SUV_ratio": [0.702, 0.828], "Smoking": [1, 0], "RECIST_response": ["Responder", "Non-responder"], "Histological": ["LUAD", "LUSC"], "cN": ["cN2", "cN0-1"]})
    st.download_button("Download CSV template", data=template.to_csv(index=False).encode("utf-8-sig"), file_name="pcr_prediction_template.csv", mime="text/csv")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            result_df = predict_batch(df)
            st.success(f"Prediction completed for {len(result_df)} patients.")
            st.dataframe(result_df, width="stretch", hide_index=True)
            st.download_button("Download prediction results", data=result_df.to_csv(index=False).encode("utf-8-sig"), file_name="pcr_prediction_results.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

with method_tab:
    st.markdown('<div class="section-label">Model information</div>', unsafe_allow_html=True)
    st.markdown("""
    **Predictors**  
    Pre-treatment SUVmax, post-treatment SUVmax, SUV reduction ratio, smoking status, RECIST response, histological subtype, and clinical N stage.

    **Primary internal validation**  
    Training-only repeated nested stratified five-fold cross-validation, repeated five times. Variable selection and hyperparameter tuning were performed inside the training data only. The validation cohort was not used for model selection.

    **Output**  
    The calculator returns an estimated pCR probability and a threshold-based classification using the aggregated outer-fold out-of-fold Youden threshold.

    **Interpretation**  
    This is a research-use prototype developed from retrospective data. Prospective and geographically diverse external validation is required before clinical implementation.
    """)
    st.markdown('<div class="footer-note">Suggested manuscript wording: An online prediction calculator was implemented based on the final Random Forest pipeline. The calculator accepts the seven selected predictors and returns an individualized pCR probability using the same preprocessing, feature order, and model parameters as the final analytical model.</div>', unsafe_allow_html=True)
