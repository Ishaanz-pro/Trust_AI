class Config:
    PROJECT_NAME: str = "TRUST-AI Decision Support"
    MODEL_PATH: str = "models/xgboost_loan_v1.joblib"
    CONFIDENCE_THRESHOLD: float = 0.85
    SENSITIVE_FEATURES: list = ["gender", "age_group"]
    LOG_LEVEL: str = "INFO"

settings = Config()
