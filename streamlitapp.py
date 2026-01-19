# 🔧 FIX: Ensure backend module is discoverable on Streamlit Cloud
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import base64
from datetime import datetime

from backend.services.model_service import ModelService
from backend.services.preprocessing import DataPreprocessor
from backend.services.decision_engine import DecisionEngine
from backend.services.explainability import ExplainerService

# Page configuration
st.set_page_config(
    page_title="TRUST-AI Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .decision-approve {
        color: #2ecc71;
        font-size: 2.5em;
        font-weight: bold;
    }
    .decision-decline {
        color: #e74c3c;
        font-size: 2.5em;
        font-weight: bold;
    }
    .decision-review {
        color: #f39c12;
        font-size: 2.5em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize services (cached to avoid reloading)
@st.cache_resource
def load_services():
    """Load and initialize all backend services"""
    model_svc = ModelService()
    preprocessor = DataPreprocessor()

    # Mock training data for SHAP background
    train_df = pd.DataFrame({
        'income': [5000, 2000, 8000, 50000, 120000],
        'loan_amount': [200, 100, 400, 15000, 50000],
        'credit_score': [700, 500, 800, 750, 790],
        'education': [1, 0, 1, 1, 1],
        'self_employed': [0, 1, 0, 0, 1],
        'property_area': [1, 2, 0, 1, 1]
    })

    # Fit preprocessor and train model
    train_df_processed = preprocessor.fit_transform(train_df)
    model_svc.train(train_df_processed, pd.Series([1, 0, 1, 1, 1]))

    explainer_svc = ExplainerService(
        model_svc.model,
        train_df_processed
    )

    return model_svc, preprocessor, explainer_svc


# Load services
model_svc, preprocessor, explainer_svc = load_services()

st.title("🛡️ TRUST-AI: Responsible Decision Support")
st.markdown("### Explainable AI for Loan Approval Decisions")
st.markdown("---")

# Sidebar form inputs
st.sidebar.header("📋 Application Input")
st.sidebar.markdown("Enter loan applicant details:")

income = st.sidebar.number_input("Annual Income ($)", min_value=0, value=50000, step=5000)
loan_amt = st.sidebar.number_input("Loan Amount ($)", min_value=0, value=15000, step=1000)

credit_score = st.sidebar.slider("Credit Score", 300, 850, 700, 10)

edu = st.sidebar.selectbox("Education Level", ["Graduate", "Not Graduate"])
emp = st.sidebar.selectbox("Self Employed", ["No", "Yes"])
prop = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.sidebar.markdown("---")
analyze_btn = st.sidebar.button("🔍 Analyze Application", use_container_width=True)

# Main content
if analyze_btn:
    with st.spinner("⏳ Processing application..."):
        try:
            payload = {
                "income": income,
                "loan_amount": loan_amt,
                "credit_score": credit_score,
                "education": "1" if edu == "Graduate" else "0",
                "self_employed": "1" if emp == "Yes" else "0",
                "property_area": "1" if prop == "Urban" else "2" if prop == "Semiurban" else "0"
            }

            processed_data = preprocessor.transform_single(payload)
            prob = model_svc.predict_proba(processed_data)
            decision = DecisionEngine.evaluate(prob)
            explanation_image = explainer_svc.get_local_explanation(processed_data)

            col1, col2, col3 = st.columns([1, 2, 1])

            with col2:
                st.markdown("### 📊 Final Decision")
                decision_text = decision['decision']
                confidence = decision['confidence']

                if decision_text == "APPROVE":
                    st.markdown(f"<p class='decision-approve'>✅ {decision_text}</p>", unsafe_allow_html=True)
                elif decision_text == "MANUAL_REVIEW":
                    st.markdown(f"<p class='decision-review'>⚠️ {decision_text}</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<p class='decision-decline'>❌ {decision_text}</p>", unsafe_allow_html=True)

                st.metric("Confidence Score", f"{confidence*100:.2f}%")
                st.info(f"💡 {decision['reason']}")

            st.markdown("---")
            st.markdown("### 📊 Explainability Analysis")

            img_data = base64.b64decode(explanation_image)
            st.image(img_data, use_container_width=True)

            st.success("✅ Application processed successfully!")
            st.caption(f"Processed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        except Exception as e:
            st.error(f"❌ Error: {e}")

else:
    st.markdown("### 🚀 Enter details in the sidebar and click Analyze")

    if model_svc.is_trained:
        st.success("✅ AI Model loaded and ready")
    else:
        st.error("❌ AI Model not loaded")
