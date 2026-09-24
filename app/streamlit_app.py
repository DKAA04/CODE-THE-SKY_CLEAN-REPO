"""Demo UI — upload one CSV, see prediction + spectrum."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import fft_magnitude, extract_features
from src.config import SAMPLE_RATE_HZ, PROCESSED_DIR


@st.cache_resource
def load_model():
    return joblib.load(PROCESSED_DIR / "model_gbm.joblib")

st.set_page_config(page_title="ADB Safegate - Structural Integrity", layout="wide", page_icon="P")

st.markdown("""
<style>
[data-testid="stMetricValue"] {font-size: 28px;}
[data-testid="stMetricLabel"] {font-size: 13px; color: #888;}
</style>
""", unsafe_allow_html=True)

st.info("Research prototype. Predictions require a locally trained model and an authorized dataset.")
st.caption("Historical author-reported results: 82.6% local by-board accuracy; "
           "82.0% BigQuery ML. These are not measurements of this running session. "
           "See the README for reproducibility and split limitations.")

st.divider()

st.title("Structural Integrity Monitor")
st.caption("Vibration-based bolt-condition classifier · ADB Safegate Code the Sky 2026 · github.com/DKAA04/CODE-THE-SKY_CLEAN-REPO")

uploaded = st.file_uploader("Upload a vibration capture (CSV)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    df.columns = [c.strip() for c in df.columns]
    arr = df[["X-axis", "Y-Axis", "Z-Axis"]].to_numpy(dtype=float)[:8192]

    feats = extract_features(arr)

    if not (PROCESSED_DIR / "model_gbm.joblib").exists():
        st.warning("No trained model found. Follow the README to prepare an authorized dataset and train the model locally.")
        st.stop()
    bundle = load_model()
    clf = bundle["model"]
    feature_cols = bundle["feature_cols"]
    classes = bundle["classes"]

    x = np.array([[feats.get(c, 0.0) for c in feature_cols]])
    pred = clf.predict(x)[0]
    probs = clf.predict_proba(x)[0]

    if pred == "30NM":
        color = "#1D9E75"; emoji_label = "HEALTHY"; sub = "30 N·m torque - tight joint"
    elif pred == "Loose":
        color = "#E24B4A"; emoji_label = "LOOSE BOLTS"; sub = "Resonance shifted to ~150 Hz - immediate fault"
    else:
        color = "#EF9F27"; emoji_label = "MIX-45 DEG FAULT"; sub = "Partial stiffness loss"

    confidence = probs[list(classes).index(pred)]

    st.markdown(f"""
    <div style="background: {color}22; border-left: 5px solid {color}; padding: 1.5rem;
                border-radius: 8px; margin: 1rem 0;">
      <div style="color: {color}; font-size: 13px; font-weight: 500; letter-spacing: 1px;">
        PREDICTED CONDITION
      </div>
      <div style="font-size: 36px; font-weight: 600; color: {color}; margin: 4px 0;">
        {emoji_label}
      </div>
      <div style="font-size: 14px; color: #aaa;">{sub}</div>
      <div style="font-size: 13px; color: #888; margin-top: 8px;">
        Confidence: <strong style="color: {color};">{confidence:.1%}</strong>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # === Capture fingerprint metrics ===
    st.markdown("##### Capture spectral fingerprint")
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric(
        "Dominant frequency",
        f"{feats['x_peak1_freq']:.0f} Hz",
        f"X-axis · peak {feats['x_peak1_amp']:.0f}",
    )
    fc2.metric(
        "Spectral centroid",
        f"{feats['x_centroid']:.0f} Hz",
        "Energy center of mass",
    )
    fc3.metric(
        "Half-power bandwidth",
        f"{feats['x_hpbw']:.1f} Hz",
        "Damping proxy (X-axis)",
    )
    st.divider()

    prob_df = pd.DataFrame({"class": classes, "probability": probs})
    st.bar_chart(prob_df.set_index("class"))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Time domain (303 ms snapshot)")
        t = np.arange(len(arr)) / SAMPLE_RATE_HZ * 1000  # ms
        fig_t = go.Figure()
        for i, name in enumerate(["X", "Y", "Z"]):
            fig_t.add_trace(go.Scatter(x=t, y=arr[:, i], name=name, mode="lines"))
        fig_t.update_layout(xaxis_title="ms", yaxis_title="ADC counts", height=350)
        st.plotly_chart(fig_t, use_container_width=True)

    with col2:
        st.subheader("Frequency spectrum")
        fig_f = go.Figure()
        for i, name in enumerate(["X", "Y", "Z"]):
            freqs, mag = fft_magnitude(arr[:, i])
            fig_f.add_trace(go.Scatter(x=freqs, y=mag, name=name, mode="lines"))
        fig_f.update_layout(xaxis_title="Hz", yaxis_title="Magnitude", xaxis_type="log", height=350)
        st.plotly_chart(fig_f, use_container_width=True)

    st.subheader("Physics-grounded features")
    show_feats = {
        "Spectral centroid (Hz) - X": round(feats["x_centroid"], 1),
        "Spectral centroid (Hz) - Y": round(feats["y_centroid"], 1),
        "Spectral centroid (Hz) - Z": round(feats["z_centroid"], 1),
        "Half-power bandwidth (Hz) - X (damping proxy)": round(feats["x_hpbw"], 1),
        "Top peak (Hz) - X": round(feats["x_peak1_freq"], 1),
        "Top peak (Hz) - Y": round(feats["y_peak1_freq"], 1),
        "Top peak (Hz) - Z": round(feats["z_peak1_freq"], 1),
    }
    st.table(pd.DataFrame(show_feats.items(), columns=["Feature", "Value"]))
else:
    st.info("Upload a CSV from the dataset to see classification + spectrum.")
