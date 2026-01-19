import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class ModelService:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X: 'pd.DataFrame', y: 'pd.Series') -> None:
        """Train the model on provided data"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Get class predictions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> float:
        """Get probability of approval (class 1)"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        probas = self.model.predict_proba(X)
        # Return probability of approval (class 1)
        return probas[0][1]

    def save(self, path: str) -> None:
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str) -> None:
        """Load model from disk"""
        self.model = joblib.load(path)
        self.is_trained = True
