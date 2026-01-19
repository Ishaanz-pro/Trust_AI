from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.services.model_service import ModelService
from backend.services.preprocessing import DataPreprocessor
from backend.services.decision_engine import DecisionEngine
from backend.services.explainability import ExplainerService
from backend.utils.logger import logger
import pandas as pd
import uvicorn

app = FastAPI(
    title="TRUST-AI API",
    description="Explainable AI for Loan Decision Support",
    version="1.0.0"
)

# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Internal State (In-memory for demo; use Redis/DB for prod)
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

# Fit preprocessor
train_df_processed = preprocessor.fit_transform(train_df)
model_svc.train(train_df_processed, pd.Series([1, 0, 1, 1, 1]))
explainer_svc = ExplainerService(model_svc.model, train_df_processed)

logger.info("✅ TRUST-AI Backend initialized successfully!")

class LoanApplication(BaseModel):
    income: float
    loan_amount: float
    credit_score: float
    education: str
    self_employed: str
    property_area: str

@app.get("/")
async def root():
    return {
        "message": "🛡️ TRUST-AI Decision Support Platform",
        "docs": "/docs",
        "status": "ready"
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_loan(app_data: LoanApplication):
    try:
        logger.info(f"📊 Processing loan application: income=${app_data.income}")
        
        # Preprocess
        processed_data = preprocessor.transform_single(app_data.dict())
        
        # Predict probability
        prob = model_svc.predict_proba(processed_data)
        
        # Decision
        decision = DecisionEngine.evaluate(prob)
        
        # Explanation
        explanation = explainer_svc.get_local_explanation(processed_data)
        
        logger.info(f"✅ Decision: {decision['decision']} (confidence: {decision['confidence']:.2%})")
        
        return {
            **decision,
            "explanation_image": explanation
        }
    except Exception as e:
        logger.error(f"❌ Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Inference Failed")

@app.get("/feature-importance")
async def get_feature_importance():
    """Get overall feature importance from the model"""
    return explainer_svc.get_feature_importance()

if __name__ == "__main__":
    uvicorn.run(
        "backendmain:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )