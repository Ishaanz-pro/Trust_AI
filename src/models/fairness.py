"""
Fairness Auditing Module - Disparate Impact Analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class FairnessAuditor:
    """Fairness auditing with disparate impact analysis"""
    
    def __init__(self):
        self.protected_attributes = ['gender', 'race']
        
    def calculate_disparate_impact(self, df: pd.DataFrame, 
                                   predictions: np.ndarray,
                                   protected_attr: str) -> Dict:
        """
        Calculate disparate impact ratio for a protected attribute.
        
        Disparate Impact Ratio = (Approval rate for protected group) / (Approval rate for reference group)
        A ratio < 0.8 indicates potential adverse impact (80% rule)
        
        Args:
            df: DataFrame with protected attributes
            predictions: Binary predictions (0 or 1)
            protected_attr: Name of protected attribute column
            
        Returns:
            Dictionary with disparate impact metrics
        """
        df_copy = df.copy()
        df_copy['prediction'] = predictions
        
        results = {}
        unique_values = sorted(df_copy[protected_attr].unique())
        
        # Calculate approval rates for each group
        approval_rates = {}
        group_sizes = {}
        
        for value in unique_values:
            group_mask = df_copy[protected_attr] == value
            group_data = df_copy[group_mask]
            approval_rate = group_data['prediction'].mean()
            approval_rates[value] = approval_rate
            group_sizes[value] = len(group_data)
        
        # Calculate disparate impact ratios
        # Use highest approval rate as reference
        max_rate = max(approval_rates.values())
        reference_group = [k for k, v in approval_rates.items() if v == max_rate][0]
        
        disparate_impact_ratios = {}
        for group, rate in approval_rates.items():
            if max_rate > 0:
                disparate_impact_ratios[group] = rate / max_rate
            else:
                disparate_impact_ratios[group] = 1.0
        
        results = {
            'protected_attribute': protected_attr,
            'reference_group': reference_group,
            'approval_rates': approval_rates,
            'group_sizes': group_sizes,
            'disparate_impact_ratios': disparate_impact_ratios,
            'passes_80_rule': all(ratio >= 0.8 for ratio in disparate_impact_ratios.values())
        }
        
        return results
    
    def audit_predictions(self, df: pd.DataFrame, predictions: np.ndarray) -> Dict:
        """
        Perform comprehensive fairness audit
        
        Args:
            df: DataFrame with protected attributes
            predictions: Binary predictions
            
        Returns:
            Dictionary with audit results for all protected attributes
        """
        audit_results = {
            'overall_approval_rate': predictions.mean(),
            'total_applications': len(predictions),
            'protected_attributes': {}
        }
        
        for attr in self.protected_attributes:
            if attr in df.columns:
                attr_results = self.calculate_disparate_impact(df, predictions, attr)
                audit_results['protected_attributes'][attr] = attr_results
        
        # Overall fairness assessment
        all_pass = all(
            results['passes_80_rule'] 
            for results in audit_results['protected_attributes'].values()
        )
        
        audit_results['overall_fairness'] = 'PASS' if all_pass else 'FAIL'
        
        return audit_results
    
    def generate_fairness_report(self, audit_results: Dict) -> str:
        """Generate human-readable fairness report"""
        report = []
        report.append("=" * 60)
        report.append("FAIRNESS AUDIT REPORT")
        report.append("=" * 60)
        report.append(f"\nTotal Applications: {audit_results['total_applications']}")
        report.append(f"Overall Approval Rate: {audit_results['overall_approval_rate']:.2%}")
        report.append(f"\nOverall Fairness Assessment: {audit_results['overall_fairness']}")
        report.append("\n" + "-" * 60)
        
        for attr, results in audit_results['protected_attributes'].items():
            report.append(f"\nProtected Attribute: {attr.upper()}")
            report.append(f"Reference Group: {results['reference_group']}")
            report.append(f"Passes 80% Rule: {'YES' if results['passes_80_rule'] else 'NO'}")
            report.append("\nGroup Analysis:")
            
            for group in sorted(results['approval_rates'].keys()):
                approval_rate = results['approval_rates'][group]
                di_ratio = results['disparate_impact_ratios'][group]
                size = results['group_sizes'][group]
                
                status = "✓" if di_ratio >= 0.8 else "✗"
                report.append(f"  {status} Group {group}:")
                report.append(f"      Size: {size} applications")
                report.append(f"      Approval Rate: {approval_rate:.2%}")
                report.append(f"      Disparate Impact Ratio: {di_ratio:.3f}")
            
            report.append("-" * 60)
        
        report.append("\nNote: Disparate Impact Ratio < 0.8 may indicate adverse impact")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def calculate_demographic_parity(self, df: pd.DataFrame, 
                                     predictions: np.ndarray,
                                     protected_attr: str) -> float:
        """
        Calculate demographic parity difference.
        
        Returns the maximum difference in approval rates between groups.
        A value close to 0 indicates better fairness.
        """
        df_copy = df.copy()
        df_copy['prediction'] = predictions
        
        approval_rates = []
        for value in df_copy[protected_attr].unique():
            group_mask = df_copy[protected_attr] == value
            approval_rate = df_copy[group_mask]['prediction'].mean()
            approval_rates.append(approval_rate)
        
        return max(approval_rates) - min(approval_rates)
    
    def calculate_equal_opportunity_difference(self, df: pd.DataFrame,
                                               predictions: np.ndarray,
                                               true_labels: np.ndarray,
                                               protected_attr: str) -> float:
        """
        Calculate equal opportunity difference (difference in true positive rates).
        
        Measures whether qualified applicants have equal opportunity across groups.
        """
        df_copy = df.copy()
        df_copy['prediction'] = predictions
        df_copy['true_label'] = true_labels
        
        # Calculate TPR for each group (among truly qualified applicants)
        tpr_rates = []
        for value in df_copy[protected_attr].unique():
            group_mask = df_copy[protected_attr] == value
            qualified_mask = df_copy['true_label'] == 1
            combined_mask = group_mask & qualified_mask
            
            if combined_mask.sum() > 0:
                tpr = df_copy[combined_mask]['prediction'].mean()
                tpr_rates.append(tpr)
        
        if len(tpr_rates) < 2:
            return 0.0
        
        return max(tpr_rates) - min(tpr_rates)


def audit_model_fairness(model, test_data: pd.DataFrame, true_labels: np.ndarray = None):
    """
    Convenience function to audit model fairness
    
    Args:
        model: Trained model with predict method
        test_data: DataFrame with features and protected attributes
        true_labels: Optional true labels for equal opportunity analysis
    """
    auditor = FairnessAuditor()
    
    # Get predictions
    predictions = model.predict(test_data.drop('approved', axis=1, errors='ignore'))
    
    # Run audit
    audit_results = auditor.audit_predictions(test_data, predictions)
    
    # Generate and print report
    report = auditor.generate_fairness_report(audit_results)
    print(report)
    
    # Additional metrics if true labels provided
    if true_labels is not None:
        print("\nAdditional Fairness Metrics:")
        for attr in auditor.protected_attributes:
            if attr in test_data.columns:
                dp_diff = auditor.calculate_demographic_parity(test_data, predictions, attr)
                eo_diff = auditor.calculate_equal_opportunity_difference(
                    test_data, predictions, true_labels, attr
                )
                print(f"\n{attr.upper()}:")
                print(f"  Demographic Parity Difference: {dp_diff:.4f}")
                print(f"  Equal Opportunity Difference: {eo_diff:.4f}")
    
    return audit_results


if __name__ == "__main__":
    # Example usage
    from loan_model import LoanApprovalModel
    
    model = LoanApprovalModel()
    df = model.create_sample_data(n_samples=1000)
    
    X = df.drop('approved', axis=1)
    y = df['approved']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.train(X_train, y_train)
    
    # Audit fairness
    test_df = X_test.copy()
    audit_model_fairness(model, test_df, y_test.values)
