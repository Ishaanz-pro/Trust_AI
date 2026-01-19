"""
Streamlit UI for Trust AI Platform
"""
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict

# API Configuration
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Trust AI - Ethical Loan Approval",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .approved {
        color: #28a745;
        font-weight: bold;
    }
    .rejected {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def plot_shap_explanation(explanation: Dict[str, float], title="SHAP Feature Impact"):
    """Create SHAP explanation visualization"""
    features = list(explanation.keys())
    values = list(explanation.values())
    
    # Sort by absolute value
    sorted_indices = np.argsort([abs(v) for v in values])[::-1]
    features = [features[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    colors = ['#d62728' if v < 0 else '#2ca02c' for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{v:+.4f}" for v in values],
        textposition='auto',
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="SHAP Value (Impact on Prediction)",
        yaxis_title="Features",
        height=400,
        showlegend=False
    )
    
    return fig


def plot_feature_importance(importance: Dict[str, float]):
    """Create feature importance visualization"""
    features = list(importance.keys())
    values = list(importance.values())
    
    # Sort by value
    sorted_indices = np.argsort(values)[::-1]
    features = [features[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker=dict(color='#1f77b4'),
    ))
    
    fig.update_layout(
        title="Model Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Features",
        height=400
    )
    
    return fig


def plot_fairness_metrics(audit_results: Dict):
    """Visualize fairness audit results"""
    figures = []
    
    for attr, results in audit_results['protected_attributes'].items():
        groups = list(results['approval_rates'].keys())
        approval_rates = [results['approval_rates'][g] for g in groups]
        di_ratios = [results['disparate_impact_ratios'][g] for g in groups]
        
        # Approval rates chart
        fig1 = go.Figure(go.Bar(
            x=[f"Group {g}" for g in groups],
            y=approval_rates,
            marker=dict(color='#1f77b4'),
            text=[f"{r:.1%}" for r in approval_rates],
            textposition='auto',
        ))
        
        fig1.update_layout(
            title=f"Approval Rates by {attr.capitalize()}",
            yaxis_title="Approval Rate",
            height=300
        )
        
        # Disparate impact ratios
        colors = ['#28a745' if r >= 0.8 else '#dc3545' for r in di_ratios]
        fig2 = go.Figure(go.Bar(
            x=[f"Group {g}" for g in groups],
            y=di_ratios,
            marker=dict(color=colors),
            text=[f"{r:.3f}" for r in di_ratios],
            textposition='auto',
        ))
        
        fig2.add_hline(y=0.8, line_dash="dash", line_color="red", 
                      annotation_text="80% Rule Threshold")
        
        fig2.update_layout(
            title=f"Disparate Impact Ratios - {attr.capitalize()}",
            yaxis_title="Ratio",
            height=300
        )
        
        figures.extend([fig1, fig2])
    
    return figures


def main():
    # Header
    st.markdown('<div class="main-header">🤖 Trust AI Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ethical AI Decision Support for Loan Approvals</div>', 
                unsafe_allow_html=True)
    
    # Check API status
    if not check_api_health():
        st.error("⚠️ API is not running. Please start the API server first.")
        st.code("python src/api/main.py", language="bash")
        return
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Single Prediction", "Batch Analysis", "Fairness Audit", "Model Info"]
    )
    
    if page == "Single Prediction":
        show_single_prediction()
    elif page == "Batch Analysis":
        show_batch_analysis()
    elif page == "Fairness Audit":
        show_fairness_audit()
    elif page == "Model Info":
        show_model_info()


def show_single_prediction():
    """Single loan application prediction page"""
    st.header("Single Loan Application Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Applicant Information")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        income = st.number_input("Annual Income ($)", min_value=0, value=60000, step=1000)
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
        loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=25000, step=1000)
    
    with col2:
        st.subheader("Additional Details")
        employment_length = st.number_input("Employment Length (years)", min_value=0, value=5)
        debt_to_income = st.slider("Debt to Income Ratio", min_value=0.0, max_value=1.0, 
                                   value=0.3, step=0.01)
        num_credit_lines = st.number_input("Number of Credit Lines", min_value=0, value=5)
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        race = st.selectbox("Race Category", options=[0, 1, 2, 3])
    
    if st.button("Analyze Application", type="primary"):
        with st.spinner("Analyzing..."):
            # Prepare request
            application = {
                "age": age,
                "income": income,
                "credit_score": credit_score,
                "loan_amount": loan_amount,
                "employment_length": employment_length,
                "debt_to_income": debt_to_income,
                "num_credit_lines": num_credit_lines,
                "gender": gender,
                "race": race
            }
            
            # Make prediction
            response = requests.post(f"{API_URL}/predict", json=application)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display result
                st.subheader("Prediction Result")
                
                col1, col2 = st.columns(2)
                with col1:
                    decision = "APPROVED ✅" if result['approved'] else "REJECTED ❌"
                    decision_class = "approved" if result['approved'] else "rejected"
                    st.markdown(f'<p class="{decision_class}" style="font-size: 2rem;">{decision}</p>', 
                              unsafe_allow_html=True)
                
                with col2:
                    st.metric("Approval Probability", 
                            f"{result['approval_probability']:.1%}")
                
                # SHAP Explanation
                st.subheader("Explainability - Feature Impact (SHAP)")
                st.plotly_chart(plot_shap_explanation(result['explanation']), 
                              use_container_width=True)
                
                st.info("🔍 **SHAP values** show how each feature influenced this specific decision. "
                       "Positive values (green) push toward approval, negative values (red) push toward rejection.")
                
                # Feature Importance
                st.subheader("Model Feature Importance")
                st.plotly_chart(plot_feature_importance(result['feature_importance']), 
                              use_container_width=True)
            else:
                st.error(f"Error: {response.text}")


def show_batch_analysis():
    """Batch analysis page"""
    st.header("Batch Loan Application Analysis")
    
    st.info("Upload a CSV file with loan applications or use sample data")
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data Preview")
        st.dataframe(df.head(10))
        
        if st.button("Analyze Batch", type="primary"):
            with st.spinner("Processing batch..."):
                applications = df.to_dict('records')
                response = requests.post(f"{API_URL}/predict/batch", 
                                       json={"applications": applications})
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success(f"Processed {result['total_applications']} applications")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Applications", result['total_applications'])
                    with col2:
                        st.metric("Approval Rate", f"{result['approval_rate']:.1%}")
                    with col3:
                        approved = int(result['total_applications'] * result['approval_rate'])
                        st.metric("Approved", approved)
                    
                    # Show detailed results
                    st.subheader("Detailed Results")
                    predictions_df = pd.DataFrame(result['predictions'])
                    st.dataframe(predictions_df)
                else:
                    st.error(f"Error: {response.text}")
    else:
        st.write("No file uploaded. Generate sample data:")
        if st.button("Generate Sample Data"):
            # Generate sample applications
            np.random.seed(42)
            n_samples = 100
            
            sample_data = {
                'age': np.random.randint(18, 70, n_samples),
                'income': np.random.randint(20000, 200000, n_samples),
                'credit_score': np.random.randint(300, 850, n_samples),
                'loan_amount': np.random.randint(5000, 100000, n_samples),
                'employment_length': np.random.randint(0, 30, n_samples),
                'debt_to_income': np.random.uniform(0, 1, n_samples),
                'num_credit_lines': np.random.randint(0, 20, n_samples),
                'gender': np.random.choice([0, 1], n_samples),
                'race': np.random.choice([0, 1, 2, 3], n_samples),
            }
            
            df = pd.DataFrame(sample_data)
            st.session_state['sample_data'] = df
            st.dataframe(df.head(10))
            
            # Analyze button
            if st.button("Analyze Sample Data", type="primary"):
                with st.spinner("Processing..."):
                    applications = df.to_dict('records')
                    response = requests.post(f"{API_URL}/predict/batch", 
                                           json={"applications": applications})
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Applications", result['total_applications'])
                        with col2:
                            st.metric("Approval Rate", f"{result['approval_rate']:.1%}")
                        with col3:
                            approved = int(result['total_applications'] * result['approval_rate'])
                            st.metric("Approved", approved)


def show_fairness_audit():
    """Fairness audit page"""
    st.header("Fairness Audit")
    
    st.info("Analyze fairness metrics including disparate impact analysis")
    
    if st.button("Generate Test Data & Run Audit", type="primary"):
        with st.spinner("Generating data and running audit..."):
            # Generate sample applications
            np.random.seed(42)
            n_samples = 500
            
            sample_data = {
                'age': np.random.randint(18, 70, n_samples),
                'income': np.random.randint(20000, 200000, n_samples),
                'credit_score': np.random.randint(300, 850, n_samples),
                'loan_amount': np.random.randint(5000, 100000, n_samples),
                'employment_length': np.random.randint(0, 30, n_samples),
                'debt_to_income': np.random.uniform(0, 1, n_samples),
                'num_credit_lines': np.random.randint(0, 20, n_samples),
                'gender': np.random.choice([0, 1], n_samples),
                'race': np.random.choice([0, 1, 2, 3], n_samples),
            }
            
            df = pd.DataFrame(sample_data)
            applications = df.to_dict('records')
            
            # Run audit
            response = requests.post(f"{API_URL}/audit", 
                                   json={"applications": applications})
            
            if response.status_code == 200:
                result = response.json()
                
                # Overall metrics
                st.subheader("Overall Fairness Assessment")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Applications", result['total_applications'])
                with col2:
                    st.metric("Overall Approval Rate", 
                            f"{result['overall_approval_rate']:.1%}")
                with col3:
                    fairness_color = "🟢" if result['overall_fairness'] == "PASS" else "🔴"
                    st.metric("Fairness Status", 
                            f"{fairness_color} {result['overall_fairness']}")
                
                # Visualizations
                st.subheader("Fairness Metrics Visualization")
                figures = plot_fairness_metrics(result)
                
                cols = st.columns(2)
                for i, fig in enumerate(figures):
                    with cols[i % 2]:
                        st.plotly_chart(fig, use_container_width=True)
                
                # Detailed report
                st.subheader("Detailed Fairness Report")
                st.text(result['report'])
            else:
                st.error(f"Error: {response.text}")


def show_model_info():
    """Model information page"""
    st.header("Model Information")
    
    response = requests.get(f"{API_URL}/model/info")
    
    if response.status_code == 200:
        info = response.json()
        
        st.subheader("Model Details")
        st.write(f"**Model Type:** {info['model_type']}")
        
        st.subheader("Features Used")
        st.write(info['features'])
        
        st.subheader("Feature Importance")
        st.plotly_chart(plot_feature_importance(info['feature_importance']), 
                      use_container_width=True)
        
        st.subheader("About Trust AI")
        st.markdown("""
        ### Key Features:
        
        - **XGBoost Predictions**: State-of-the-art gradient boosting for accurate loan approval predictions
        - **SHAP Explainability**: Understand exactly why each decision was made
        - **Fairness Auditing**: Automated disparate impact analysis to ensure fair treatment
        - **Modular Architecture**: FastAPI backend with Streamlit frontend
        
        ### Fairness Metrics:
        
        - **Disparate Impact Ratio**: Measures approval rate parity across protected groups
        - **80% Rule**: Industry standard threshold (ratio ≥ 0.8 indicates fairness)
        - **Demographic Parity**: Ensures equal approval rates across demographics
        
        ### Technology Stack:
        
        - Python, XGBoost, SHAP
        - FastAPI, Streamlit
        - scikit-learn, pandas, numpy
        
        ---
        *Trust AI - Ensuring transparency and bias mitigation in AI-driven decisions*
        """)
    else:
        st.error(f"Error fetching model info: {response.text}")


if __name__ == "__main__":
    main()
