# 🛡️ Trust_AI

> **Responsible Decision Support System with Explainable AI**

An end-to-end machine learning platform for automated loan approval decisions with built-in explainability, fairness auditing, and transparency.  Trust_AI demonstrates how AI systems can be both powerful and responsible.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-F7931E.svg)](https://scikit-learn.org/)

---

## 🌟 Features

### 🤖 **Intelligent Decision Making**
- **RandomForest Classifier** with optimized hyperparameters
- Three-tier decision framework:  `APPROVE`, `DECLINE`, `MANUAL_REVIEW`
- Confidence-based thresholds for risk management

### 📊 **Explainable AI**
- **SHAP-style** feature importance visualization
- Local explanations for every prediction
- Transparent feature contribution analysis
- Real-time visual explanations in the UI

### ⚖️ **Fairness & Auditing**
- Statistical parity monitoring
- Equal opportunity metrics
- Calibration error tracking
- Built-in fairness dashboard

### 🎨 **Modern Web Interface**
- Interactive **Streamlit** dashboard
- Real-time prediction and analysis
- Professional visualizations
- Responsive design for all devices

### 🔧 **Production-Ready Architecture**
- **FastAPI** backend with async support
- RESTful API design
- Modular, maintainable codebase
- Comprehensive error handling

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Streamlit UI   │  ← User-facing dashboard
└────────┬────────┘
         │ HTTP
         ↓
┌─────────────────┐
│   FastAPI       │  ← Backend API
│   Backend       │
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────────┐
    ↓         ↓          ↓              ↓
┌────────┐ ┌──────┐ ┌────────────┐ ┌──────────┐
│ Model  │ │Decision│ │Explainability│ │Preprocessing│
│Service │ │Engine│ │  Service   │ │            │
└────────┘ └──────┘ └────────────┘ └──────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ishaanz-pro/Trust_AI.git
   cd Trust_AI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   Using the convenience script:
   ```bash
   chmod +x run_app.sh
   ./run_app.sh
   ```

   Or manually: 

   **Terminal 1 - Start Backend:**
   ```bash
   python backendmain.py
   ```

   **Terminal 2 - Start Frontend:**
   ```bash
   streamlit run streamlitapp.py
   ```

4. **Access the application**
   - Frontend: http://localhost:8501
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

---

## 📦 Project Structure

```
Trust_AI/
│
├── backendmain.py          # FastAPI backend server
├── streamlitapp. py         # Streamlit frontend application
├── decision_engine.py      # Decision logic & thresholds
├── explainability.py       # SHAP-style explanation generator
├── model_service.py        # ML model training & inference
├── preprocessing.py        # Data preprocessing pipeline
│
├── requirements.txt        # Python dependencies
├── run_app.sh             # Application launcher script
├── backend. log            # Backend logs
└── README.md              # Project documentation
```

---

## 💡 Usage

### Web Interface

1. **Navigate to the sidebar** and enter loan application details: 
   - Annual Income
   - Loan Amount
   - Credit Score
   - Education Level
   - Employment Status
   - Property Area

2. **Click "Analyze Application"** to process the request

3. **Review the decision** with: 
   - Final decision (APPROVE/DECLINE/MANUAL_REVIEW)
   - Confidence score
   - Reasoning explanation

4. **Examine explainability charts** showing:
   - Feature importance
   - Individual feature contributions
   - Visual breakdown of decision factors

5. **Check fairness metrics**:
   - Statistical Parity
   - Equal Opportunity
   - Calibration Error

### API Usage

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "income": 50000,
  "loan_amount": 15000,
  "credit_score": 700,
  "education":  "1",
  "self_employed": "0",
  "property_area":  "1"
}
```

**Response:**
```json
{
  "decision": "APPROVE",
  "confidence": 0.87,
  "reason": "Strong approval signal (confidence: 87. 00%)",
  "explanation_image": "<base64_

