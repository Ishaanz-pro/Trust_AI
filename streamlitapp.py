import streamlit as st
import pandas as pd
import numpy as np
import base64
from datetime import datetime

from model_service import ModelService
from preprocessing import DataPreprocessor
from decision_engine import DecisionEngine
from explainability import ExplainerService

st.set_page_config(
    page_title="TRUST-AI Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.decision-approve { color: #2ecc71; font-size: 2.2em; font-weight: bold; }
.decision-decline { color: #e74c3c; font-size: 2.2em; font-weight: bold; }
.decision-review { color: #f39c12; font-size: 2.2em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_services():
    model = ModelService()
    preprocessor = DataPreprocessor()

    train_df = pd.DataFrame({
        "income": [5000, 2000, 8000, 50000, 120000],
        "loan_amount": [200, 100, 400, 15000, 50000],
        "credit_score": [700, 500, 800, 750, 790],
        "education": [1, 0, 1, 1, 1],
        "self_employed": [0, 1, 0, 0, 1],
        "property_area": [1, 2, 0, 1, 1]
    })

    X = preprocessor.fit_transform(train_df)
    y = pd.Series([1, 0, 1, 1, 1])

    model.train(X, y)
    explainer = ExplainerService(model.model, X)

    return model, preprocessor, explainer


model_svc, preprocessor, explainer_svc = load_services()

st.title("🛡️ TRUST-AI Loan Decision System")

st.sidebar.header("📋 Application Input")

income = st.sidebar.number_input("Annual Income", 0, 200000, 50000, 5000)
loan_amt = st.sidebar.number_input("Loan Amount", 0, 100000, 15000, 1000)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 700, 10)
edu = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
emp = st.sidebar.selectbox("Self Employed", ["No", "Yes"])
prop = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.sidebar.button("🔍 Analyze Application", use_container_width=True):
    payload = {
        "income": income,
        "loan_amount": loan_amt,
        "credit_score": credit_score,
        "education": 1 if edu == "Graduate" else 0,
        "self_employed": 1 if emp == "Yes" else 0,
        "property_area": 1 if prop == "Urban" else 2 if prop == "Semiurban" else 0
    }

    X_input = preprocessor.transform_single(payload)
    prob = model_svc.predict_proba(X_input)
    decision = DecisionEngine.evaluate(prob)
    explanation_img = explainer_svc.get_local_explanation(X_input)

    st.subheader("📊 Final Decision")

    if decision["decision"] == "APPROVE":
        st.markdown(f"<p class='decision-approve'>✅ APPROVED</p>", unsafe_allow_html=True)
    elif decision["decision"] == "MANUAL_REVIEW":
        st.markdown(f"<p class='decision-review'>⚠️ MANUAL REVIEW</p>", unsafe_allow_html=True)
    else:
        st.markdown(f"<p class='decision-decline'>❌ DECLINED</p>", unsafe_allow_html=True)

    st.metric("Confidence", f"{decision['confidence']*100:.2f}%")
    st.info(decision["reason"])

    st.subheader("📈 Explainability")
    st.image(base64.b64decode(explanation_img), use_container_width=True)

    st.caption(f"Processed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
