# Trust AI - Ethical AI Decision Support Platform

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

An ethical AI decision support platform for loan approvals featuring XGBoost predictions, SHAP explainability, fairness auditing, and modular FastAPI/Streamlit architecture. Designed to ensure transparency and bias mitigation in AI-driven lending decisions.

## 🌟 Features

- **XGBoost Predictions**: State-of-the-art gradient boosting model for accurate loan approval predictions
- **SHAP Explainability**: Understand exactly why each decision was made with feature-level explanations
- **Fairness Auditing**: Automated disparate impact analysis to ensure equitable treatment across demographics
- **Modular Architecture**: Clean separation between FastAPI backend and Streamlit frontend
- **RESTful API**: Comprehensive API for integration with existing systems
- **Interactive UI**: User-friendly Streamlit interface for data scientists and loan officers

## 🏗️ Architecture

```
Trust_AI/
├── src/
│   ├── models/
│   │   ├── loan_model.py      # XGBoost model with SHAP
│   │   └── fairness.py         # Fairness auditing module
│   ├── api/
│   │   └── main.py             # FastAPI backend
│   └── ui/
│       └── app.py              # Streamlit frontend
├── data/                        # Sample data directory
├── models/                      # Trained models directory
├── requirements.txt             # Python dependencies
├── run_app.sh                   # Application startup script
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ishaanz-pro/Trust_AI.git
   cd Trust_AI
   ```

2. **Run the application**
   ```bash
   chmod +x run_app.sh
   ./run_app.sh
   ```

   This script will:
   - Create a virtual environment
   - Install all dependencies
   - Train the initial model (if needed)
   - Start the FastAPI backend on `http://localhost:8000`
   - Start the Streamlit UI on `http://localhost:8501`

3. **Access the application**
   - **Web UI**: http://localhost:8501
   - **API**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs

### Manual Setup

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train the model
python src/models/loan_model.py

# Start API (in one terminal)
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Start UI (in another terminal)
streamlit run src/ui/app.py --server.port 8501
```

## 📊 Technology Stack

- **Machine Learning**: XGBoost, scikit-learn
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit, Plotly
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn

## 🔍 Key Components

### 1. Loan Approval Model (`loan_model.py`)

- XGBoost classifier for binary loan approval decisions
- SHAP integration for model explainability
- Feature importance analysis
- Model persistence and loading

### 2. Fairness Auditing (`fairness.py`)

- **Disparate Impact Analysis**: Measures approval rate disparities across protected groups
- **80% Rule Compliance**: Industry-standard fairness threshold
- **Demographic Parity**: Ensures equal treatment across demographics
- **Equal Opportunity**: Evaluates true positive rate parity

### 3. FastAPI Backend (`api/main.py`)

RESTful API endpoints:
- `POST /predict` - Single loan application prediction
- `POST /predict/batch` - Batch predictions
- `POST /audit` - Fairness audit on batch predictions
- `GET /model/info` - Model information and feature importance
- `GET /health` - Health check

### 4. Streamlit UI (`ui/app.py`)

Interactive interface with:
- Single application analysis with SHAP explanations
- Batch analysis for multiple applications
- Fairness audit dashboard with visualizations
- Model information and feature importance

## 🎯 Use Cases

1. **Loan Officers**: Review individual applications with AI assistance and explainability
2. **Risk Analysts**: Batch process applications and analyze approval patterns
3. **Compliance Teams**: Audit model fairness and ensure regulatory compliance
4. **Data Scientists**: Understand model behavior and feature importance

## 📈 Fairness Metrics

### Disparate Impact Ratio
Ratio of approval rates between protected and reference groups. A ratio ≥ 0.8 indicates fairness (80% rule).

### Demographic Parity
Measures the maximum difference in approval rates across demographic groups.

### Equal Opportunity
Evaluates whether qualified applicants have equal approval chances across groups.

## 🔒 Ethical Considerations

This platform is designed with ethics at its core:

- **Transparency**: SHAP explanations for every decision
- **Fairness**: Automated bias detection and reporting
- **Accountability**: Comprehensive audit trails
- **Privacy**: Minimal data collection and secure processing

## 📝 API Examples

### Single Prediction

```python
import requests

application = {
    "age": 35,
    "income": 75000,
    "credit_score": 720,
    "loan_amount": 25000,
    "employment_length": 5,
    "debt_to_income": 0.3,
    "num_credit_lines": 5,
    "gender": 0,
    "race": 1
}

response = requests.post("http://localhost:8000/predict", json=application)
result = response.json()

print(f"Approved: {result['approved']}")
print(f"Probability: {result['approval_probability']:.2%}")
print(f"Top factors: {result['explanation']}")
```

### Fairness Audit

```python
import requests

applications = [...]  # List of loan applications

response = requests.post(
    "http://localhost:8000/audit",
    json={"applications": applications}
)

audit_result = response.json()
print(audit_result['report'])
```

## 🧪 Testing

```bash
# Run model training
python src/models/loan_model.py

# Test fairness module
python src/models/fairness.py

# Test API (requires running server)
curl http://localhost:8000/health
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **SHAP**: For model explainability framework
- **XGBoost**: For the powerful gradient boosting library
- **FastAPI**: For the modern web framework
- **Streamlit**: for the interactive UI framework

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Trust AI** - Ensuring transparency and bias mitigation in AI-driven decisions 🤖✨
