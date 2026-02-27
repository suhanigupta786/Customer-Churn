import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Telco Churn Intelligence",
    layout="wide",
    page_icon="📡"
)

# =========================
# CUSTOM THEME (Telecom Dark)
# =========================
st.markdown("""
<style>
html, body, [class*="css"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: white;
}

.block-container {
    padding-top: 2rem;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 20px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 20px rgba(0,0,0,0.4);
}

.stButton>button {
    background: linear-gradient(90deg, #ff0080, #7928ca);
    color: white;
    font-weight: bold;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    border: none;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = joblib.load("churn_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# =========================
# HEADER
# =========================
st.title("📡 Telco Churn Intelligence Platform")
st.caption("AI-powered churn risk detection & retention analytics")

st.markdown("---")

# =========================
# INPUT SECTION
# =========================
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📊 Customer Profile")
    tenure = st.slider("Tenure (Months)", 0, 72, 24)
    monthly_charges = st.slider("Monthly Charges", 10.0, 120.0, 65.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 2000.0)

with col2:
    st.markdown("### 🌐 Service Configuration")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

st.markdown("---")

# =========================
# PREDICTION BUTTON
# =========================
if st.button("🚀 Run Churn Risk Analysis"):

    input_dict = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Charge_per_Tenure": monthly_charges / (tenure + 1),
        "Contract_" + contract: 1,
        "InternetService_" + internet_service: 1,
        "PaperlessBilling_Yes": 1 if paperless == "Yes" else 0
    }

    input_df = pd.DataFrame([input_dict])

    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]

    prob = model.predict_proba(input_df)[0][1]
    threshold = 0.45
    prediction = 1 if prob > threshold else 0

    # =========================
    # KPI SECTION
    # =========================
    st.markdown("## 🔎 Executive Risk Summary")

    k1, k2, k3 = st.columns(3)

    k1.metric("Churn Probability", f"{prob*100:.2f}%")

    if prediction == 1:
        k2.metric("Prediction", "Churn Likely 🔴")
    else:
        k2.metric("Prediction", "Retention Likely 🟢")

    if prob > 0.7:
        risk = "High Risk"
    elif prob > 0.4:
        risk = "Moderate Risk"
    else:
        risk = "Low Risk"

    k3.metric("Risk Level", risk)

    st.markdown("---")

    # =========================
    # RADIAL GAUGE
    # =========================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Churn Risk Meter (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#ff0080"},
            'steps': [
                {'range': [0, 40], 'color': "#00c6ff"},
                {'range': [40, 70], 'color': "#7928ca"},
                {'range': [70, 100], 'color': "#ff4b5c"}
            ]
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # =========================
    # FEATURE IMPACT
    # =========================
    st.markdown("## 📈 Key Churn Drivers")

    importances = model.feature_importances_
    indices = np.argsort(importances)[-8:]

    feature_df = pd.DataFrame({
        "Feature": np.array(model_columns)[indices],
        "Importance": importances[indices]
    }).sort_values(by="Importance", ascending=True)

    fig2 = px.bar(
        feature_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#00c6ff", "#7928ca", "#ff0080"]
    )

    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # =========================
    # STRATEGIC INSIGHTS
    # =========================
    st.markdown("## 💡 Retention Strategy Recommendation")

    if prob > 0.7:
        st.error("Immediate action required: Provide loyalty discount, contract upgrade offer, or retention outreach.")
    elif prob > 0.4:
        st.warning("Customer requires engagement monitoring. Recommend targeted promotional incentives.")
    else:
        st.success("Customer is stable. Focus on satisfaction and upsell opportunities.")