"""
Model Evaluation Module for Credit Card Fraud Detection
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef
)

import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation class
    """
    
    def __init__(self, model, X_test, y_test, model_name='Model'):
        """
        Initialize evaluator
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        
        # Get predictions
        self.y_pred = model.predict(X_test)
        
        # Get probabilities if available
        if hasattr(model, 'predict_proba'):
            self.y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            self.y_pred_proba = self.y_pred
    
    def print_classification_report(self):
        """Print detailed classification report"""
        print(f"\n{'='*60}")
        print(f"Classification Report - {self.model_name}")
        print(f"{'='*60}")
        print(classification_report(self.y_test, self.y_pred, 
                                   target_names=['Legitimate', 'Fraud']))
        print(f"{'='*60}")
    
    def plot_confusion_matrix(self, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            save_path: Optional path to save the figure
        """
        cm = confusion_matrix(self.y_test, self.y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'],
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {self.model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print confusion matrix breakdown
        tn, fp, fn, tp = cm.ravel()
        print(f"\nConfusion Matrix Breakdown:")
        print(f"True Negatives (TN): {tn:,}")
        print(f"False Positives (FP): {fp:,}")
        print(f"False Negatives (FN): {fn:,}")
        print(f"True Positives (TP): {tp:,}")
        
        return cm
    
    def plot_roc_curve(self, save_path=None):
        """
        Plot ROC curve
        
        Args:
            save_path: Optional path to save the figure
        """
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {self.model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fpr, tpr, roc_auc
    
    def plot_precision_recall_curve(self, save_path=None):
        """
        Plot Precision-Recall curve
        
        Args:
            save_path: Optional path to save the figure
        """
        precision, recall, thresholds = precision_recall_curve(
            self.y_test, self.y_pred_proba
        )
        ap_score = average_precision_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkgreen', lw=2,
                label=f'PR curve (AP = {ap_score:.4f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve - {self.model_name}', 
                 fontsize=14, fontweight='bold')
        plt.legend(loc="upper right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return precision, recall, ap_score
    
    def plot_threshold_analysis(self):
        """Plot metrics across different thresholds"""
        thresholds = np.linspace(0, 1, 100)
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (self.y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            try:
                precision = precision_score(self.y_test, y_pred_thresh, zero_division=0)
                recall = recall_score(self.y_test, y_pred_thresh, zero_division=0)
                f1 = f1_score(self.y_test, y_pred_thresh, zero_division=0)
            except:
                precision, recall, f1 = 0, 0, 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(f'Metrics vs Threshold - {self.model_name}', 
                 fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Find optimal threshold for F1-score
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        print(f"\nOptimal Threshold (for F1-Score): {optimal_threshold:.4f}")
        print(f"F1-Score at optimal threshold: {optimal_f1:.4f}")
        
        return optimal_threshold
    
    def get_all_metrics(self):
        """Calculate and return all evaluation metrics"""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, roc_auc_score, matthews_corrcoef
        )
        
        metrics = {
            'Model': self.model_name,
            'Accuracy': accuracy_score(self.y_test, self.y_pred),
            'Precision': precision_score(self.y_test, self.y_pred),
            'Recall': recall_score(self.y_test, self.y_pred),
            'F1-Score': f1_score(self.y_test, self.y_pred),
            'ROC-AUC': roc_auc_score(self.y_test, self.y_pred_proba),
            'PR-AUC': average_precision_score(self.y_test, self.y_pred_proba),
            'MCC': matthews_corrcoef(self.y_test, self.y_pred)
        }
        
        return metrics
    
    def comprehensive_evaluation(self, save_dir=None):
        """
        Run comprehensive evaluation with all plots and metrics
        
        Args:
            save_dir: Optional directory to save figures
        """
        print(f"\n{'='*70}")
        print(f"Comprehensive Evaluation: {self.model_name}")
        print(f"{'='*70}")
        
        # Get all metrics
        metrics = self.get_all_metrics()
        print("\nPerformance Metrics:")
        for key, value in metrics.items():
            if key != 'Model':
                print(f"  {key}: {value:.4f}")
        
        # Classification report
        self.print_classification_report()
        
        # Confusion matrix
        if save_dir:
            cm_path = f"{save_dir}/confusion_matrix_{self.model_name.replace(' ', '_')}.png"
        else:
            cm_path = None
        self.plot_confusion_matrix(save_path=cm_path)
        
        # ROC curve
        if save_dir:
            roc_path = f"{save_dir}/roc_curve_{self.model_name.replace(' ', '_')}.png"
        else:
            roc_path = None
        self.plot_roc_curve(save_path=roc_path)
        
        # Precision-Recall curve
        if save_dir:
            pr_path = f"{save_dir}/pr_curve_{self.model_name.replace(' ', '_')}.png"
        else:
            pr_path = None
        self.plot_precision_recall_curve(save_path=pr_path)
        
        # Threshold analysis
        optimal_threshold = self.plot_threshold_analysis()
        
        print(f"\n{'='*70}")
        print("Evaluation Complete!")
        print(f"{'='*70}\n")
        
        return metrics, optimal_threshold


class MultiModelComparison:
    """
    Compare multiple models
    """
    
    def __init__(self, models_dict, X_test, y_test):
        """
        Initialize comparison
        
        Args:
            models_dict: Dictionary of {model_name: model}
            X_test: Test features
            y_test: Test labels
        """
        self.models_dict = models_dict
        self.X_test = X_test
        self.y_test = y_test
        self.results = []
    
    def compare_all(self):
        """Compare all models"""
        for name, model in self.models_dict.items():
            evaluator = ModelEvaluator(model, self.X_test, self.y_test, name)
            metrics = evaluator.get_all_metrics()
            self.results.append(metrics)
        
        results_df = pd.DataFrame(self.results)
        return results_df
    
    def plot_comparison(self):
        """Plot comparison of all models"""
        results_df = pd.DataFrame(self.results)
        
        metrics_to_plot = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'MCC']
        
        fig = go.Figure()
        
        for metric in metrics_to_plot:
            fig.add_trace(go.Bar(
                name=metric,
                x=results_df['Model'],
                y=results_df[metric],
                text=results_df[metric].round(4),
                textposition='auto'
            ))
        
        fig.update_layout(
            title='Model Performance Comparison',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=500,
            showlegend=True
        )
        
        fig.show()
        
        return fig


def main():
    """Main execution function"""
    import joblib
    
    # Load a trained model
    model = joblib.load('../models/xgboost.pkl')
    
    # Load test data
    X_test = pd.read_csv('../data/processed/X_test.csv')
    y_test = pd.read_csv('../data/processed/y_test.csv').values.ravel()
    
    # Evaluate
    evaluator = ModelEvaluator(model, X_test, y_test, 'XGBoost')
    metrics, threshold = evaluator.comprehensive_evaluation(
        save_dir='../reports/figures'
    )
    
    return evaluator


if __name__ == "__main__":
    main()

