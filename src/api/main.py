"""
FastAPI Backend for Trust AI Platform
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.loan_model import LoanApprovalModel
from models.fairness import FairnessAuditor

app = FastAPI(
    title="Trust AI - Ethical Loan Approval API",
    description="Ethical AI Decision Support Platform with XGBoost, SHAP, and Fairness Auditing",
    version="1.0.0"
)

# Enable CORS - NOTE: In production, replace "*" with specific allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Replace with specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None
auditor = FairnessAuditor()


class LoanApplication(BaseModel):
    """Loan application input schema"""
    age: int = Field(..., ge=18, le=100, description="Applicant age")
    income: float = Field(..., ge=0, description="Annual income")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    loan_amount: float = Field(..., ge=0, description="Requested loan amount")
    employment_length: int = Field(..., ge=0, description="Years of employment")
    debt_to_income: float = Field(..., ge=0, le=1, description="Debt to income ratio")
    num_credit_lines: int = Field(..., ge=0, description="Number of credit lines")
    gender: int = Field(..., ge=0, le=1, description="Gender (0: Female, 1: Male)")
    race: int = Field(..., ge=0, le=3, description="Race category (0-3)")


class PredictionResponse(BaseModel):
    """Prediction response schema"""
    approved: bool
    approval_probability: float
    explanation: Dict[str, float]
    feature_importance: Dict[str, float]


class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema"""
    applications: List[LoanApplication]


class FairnessAuditResponse(BaseModel):
    """Fairness audit response schema"""
    overall_approval_rate: float
    total_applications: int
    overall_fairness: str
    protected_attributes: Dict
    report: str


def load_model_on_startup():
    """Load model on startup - called during application initialization"""
    global model
    model = LoanApprovalModel()
    
    model_path = 'models/loan_model.pkl'
    if os.path.exists(model_path):
        model.load(model_path)
        print(f"Model loaded from {model_path}")
    else:
        print("No pre-trained model found. Training new model...")
        from models.loan_model import train_and_save_model
        model = train_and_save_model(output_path=model_path)
        print("New model trained and saved")

# Load model at module initialization (will run when the module is imported)
load_model_on_startup()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Trust AI - Ethical Loan Approval API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "Single loan application prediction",
            "/predict/batch": "Batch loan application predictions",
            "/audit": "Fairness audit for batch predictions",
            "/health": "Health check",
            "/model/info": "Model information"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": "XGBoost Classifier",
        "features": model.feature_names,
        "feature_importance": model.get_feature_importance()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(application: LoanApplication):
    """
    Predict loan approval for a single application with SHAP explainability
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert application to DataFrame
        app_dict = application.dict()
        df = pd.DataFrame([app_dict])
        
        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]
        
        # Get SHAP explanation
        shap_values = model.explain(df)
        explanation = {}
        for i, feature in enumerate(model.feature_names):
            explanation[feature] = float(shap_values[0][i])
        
        # Get feature importance
        feature_importance = model.get_feature_importance()
        
        return PredictionResponse(
            approved=bool(prediction),
            approval_probability=float(probability[1]),
            explanation=explanation,
            feature_importance=feature_importance
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict loan approvals for multiple applications
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert applications to DataFrame
        apps_data = [app.dict() for app in request.applications]
        df = pd.DataFrame(apps_data)
        
        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)
        
        # Get SHAP explanations for each application
        shap_values = model.explain(df)
        
        results = []
        for i in range(len(predictions)):
            explanation = {}
            for j, feature in enumerate(model.feature_names):
                explanation[feature] = float(shap_values[i][j])
            
            results.append({
                "application_index": i,
                "approved": bool(predictions[i]),
                "approval_probability": float(probabilities[i][1]),
                "explanation": explanation
            })
        
        return {
            "predictions": results,
            "total_applications": len(predictions),
            "approval_rate": float(predictions.mean())
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


@app.post("/audit", response_model=FairnessAuditResponse)
async def audit_fairness(request: BatchPredictionRequest):
    """
    Perform fairness audit on batch predictions
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert applications to DataFrame
        apps_data = [app.dict() for app in request.applications]
        df = pd.DataFrame(apps_data)
        
        # Make predictions
        predictions = model.predict(df)
        
        # Perform fairness audit
        audit_results = auditor.audit_predictions(df, predictions)
        
        # Generate report
        report = auditor.generate_fairness_report(audit_results)
        
        return FairnessAuditResponse(
            overall_approval_rate=audit_results['overall_approval_rate'],
            total_applications=audit_results['total_applications'],
            overall_fairness=audit_results['overall_fairness'],
            protected_attributes=audit_results['protected_attributes'],
            report=report
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audit error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
