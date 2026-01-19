import numpy as np
import matplotlib.pyplot as plt
import base64
import io
from typing import Tuple

class ExplainerService:
    """Provide SHAP-like explanations for model decisions"""
    
    def __init__(self, model, background_data):
        self.model = model
        self.background_data = background_data
        self.feature_names = ['income', 'loan_amount', 'credit_score', 'education', 'self_employed', 'property_area']
        
    def get_local_explanation(self, instance: np.ndarray) -> str:
        """Generate SHAP-style explanation and return as base64 image"""
        # Get feature importances as a proxy for SHAP values
        importances = self.model.feature_importances_
        
        # Normalize for visualization
        normalized_importances = importances / importances.sum()
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in normalized_importances]
        
        bars = ax.barh(self.feature_names, normalized_importances, color=colors, alpha=0.8)
        ax.set_xlabel('Feature Importance Score', fontsize=12, fontweight='bold')
        ax.set_title('Feature Contribution to Loan Decision (SHAP-style)', fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(normalized_importances) * 1.2)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, normalized_importances)):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.2%}', va='center', fontsize=10)
        
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        
        return image_base64
    
    def get_feature_importance(self) -> dict:
        """Get raw feature importances"""
        return dict(zip(self.feature_names, self.model.feature_importances_))
