"""
Loan Approval Model using XGBoost with SHAP Explainability
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os


class LoanApprovalModel:
    """XGBoost-based loan approval prediction model with SHAP explainability"""
    
    def __init__(self):
        self.model = None
        self.explainer = None
        self.feature_names = None
        
    def create_sample_data(self, n_samples=1000):
        """Generate synthetic loan application data for demonstration"""
        np.random.seed(42)
        
        # Generate features
        data = {
            'age': np.random.randint(18, 70, n_samples),
            'income': np.random.randint(20000, 200000, n_samples),
            'credit_score': np.random.randint(300, 850, n_samples),
            'loan_amount': np.random.randint(5000, 100000, n_samples),
            'employment_length': np.random.randint(0, 30, n_samples),
            'debt_to_income': np.random.uniform(0, 1, n_samples),
            'num_credit_lines': np.random.randint(0, 20, n_samples),
            'gender': np.random.choice([0, 1], n_samples),  # 0: Female, 1: Male
            'race': np.random.choice([0, 1, 2, 3], n_samples),  # Simplified categories
        }
        
        df = pd.DataFrame(data)
        
        # Generate target variable with realistic logic
        approval_score = (
            (df['credit_score'] - 300) / 550 * 0.4 +
            (df['income'] - 20000) / 180000 * 0.3 +
            (1 - df['debt_to_income']) * 0.2 +
            (df['employment_length'] / 30) * 0.1
        )
        
        # Add some randomness
        approval_score += np.random.normal(0, 0.1, n_samples)
        df['approved'] = (approval_score > 0.5).astype(int)
        
        return df
    
    def train(self, X_train, y_train, X_test=None, y_test=None):
        """Train XGBoost model"""
        self.feature_names = X_train.columns.tolist()
        
        # Configure XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss'
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Initialize SHAP explainer
        self.explainer = shap.TreeExplainer(self.model)
        
        # Evaluate if test data provided
        if X_test is not None and y_test is not None:
            predictions = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            print(f"Model Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_test, predictions))
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict_proba(X)
    
    def explain(self, X):
        """Generate SHAP explanations for predictions"""
        if self.explainer is None:
            raise ValueError("Model not trained. Call train() first.")
        
        shap_values = self.explainer.shap_values(X)
        return shap_values
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        importance_dict = {}
        for feature, importance in zip(self.feature_names, self.model.feature_importances_):
            importance_dict[feature] = float(importance)
        
        return importance_dict
    
    def save(self, filepath):
        """Save model to disk"""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath):
        """Load model from disk"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.explainer = shap.TreeExplainer(self.model)
        
        return self


def train_and_save_model(output_path='models/loan_model.pkl', data_path=None):
    """Train model and save to disk"""
    model = LoanApprovalModel()
    
    # Load or create data
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df = model.create_sample_data(n_samples=2000)
        # Save sample data
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/sample_loan_data.csv', index=False)
    
    # Prepare data
    X = df.drop('approved', axis=1)
    y = df['approved']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model.train(X_train, y_train, X_test, y_test)
    
    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    print(f"\nModel saved to {output_path}")
    
    return model


if __name__ == "__main__":
    # Train and save model when run directly
    train_and_save_model()
