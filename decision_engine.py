from backend.utils.config import settings

class DecisionEngine:
    """Loan decision logic with thresholds"""
    
    APPROVAL_THRESHOLD = 0.75
    MANUAL_REVIEW_THRESHOLD = 0.50
    
    @staticmethod
    def evaluate(approval_probability: float) -> dict:
        """Make decision based on model probability"""
        if approval_probability >= DecisionEngine.APPROVAL_THRESHOLD:
            return {
                "decision": "APPROVE",
                "confidence": approval_probability,
                "reason": f"Strong approval signal (confidence: {approval_probability:.2%})"
            }
        elif approval_probability >= DecisionEngine.MANUAL_REVIEW_THRESHOLD:
            return {
                "decision": "MANUAL_REVIEW",
                "confidence": approval_probability,
                "reason": f"Borderline case, requires manual assessment (confidence: {approval_probability:.2%})"
            }
        else:
            return {
                "decision": "DECLINE",
                "confidence": 1 - approval_probability,
                "reason": f"Low approval signal (confidence: {1-approval_probability:.2%})"
            }
