"""
Utility Functions for Credit Card Fraud Detection
"""
import numpy as np
import pandas as pd
import joblib
import os
import json
from datetime import datetime


def load_model(model_path):
    """
    Load a saved model
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    if model_path.endswith('.pkl'):
        model = joblib.load(model_path)
    elif model_path.endswith('.h5'):
        try:
            from tensorflow import keras
            model = keras.models.load_model(model_path)
        except ImportError:
            raise ImportError("TensorFlow is required to load .h5 models")
    else:
        raise ValueError(f"Unsupported file format: {model_path}")
    
    return model


def save_model(model, model_path):
    """
    Save a trained model
    
    Args:
        model: Model to save
        model_path: Path to save the model
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if model_path.endswith('.pkl'):
        joblib.dump(model, model_path)
    elif model_path.endswith('.h5'):
        model.save(model_path)
    else:
        raise ValueError(f"Unsupported file format: {model_path}")
    
    print(f"Model saved to {model_path}")


def predict_transaction(model, transaction_features):
    """
    Predict if a transaction is fraudulent
    
    Args:
        model: Trained model
        transaction_features: Features of the transaction (dict or array)
        
    Returns:
        Prediction (0 or 1) and probability
    """
    # Convert dict to DataFrame if necessary
    if isinstance(transaction_features, dict):
        transaction_df = pd.DataFrame([transaction_features])
    elif isinstance(transaction_features, np.ndarray):
        transaction_df = pd.DataFrame(transaction_features)
    else:
        transaction_df = transaction_features
    
    # Get prediction
    prediction = model.predict(transaction_df)[0]
    
    # Get probability if available
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(transaction_df)[0, 1]
    else:
        probability = prediction
    
    return int(prediction), float(probability)


def calculate_fraud_score(probability):
    """
    Convert fraud probability to a risk score (0-100)
    
    Args:
        probability: Fraud probability (0-1)
        
    Returns:
        Risk score (0-100)
    """
    return int(probability * 100)


def get_risk_level(probability):
    """
    Get risk level based on fraud probability
    
    Args:
        probability: Fraud probability (0-1)
        
    Returns:
        Risk level string
    """
    if probability < 0.2:
        return "Low Risk"
    elif probability < 0.5:
        return "Medium Risk"
    elif probability < 0.8:
        return "High Risk"
    else:
        return "Critical Risk"


def create_prediction_report(transaction_features, prediction, probability, model_name="Model"):
    """
    Create a detailed prediction report
    
    Args:
        transaction_features: Transaction features
        prediction: Model prediction (0 or 1)
        probability: Fraud probability
        model_name: Name of the model used
        
    Returns:
        Report dictionary
    """
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "prediction": "Fraud" if prediction == 1 else "Legitimate",
        "fraud_probability": f"{probability:.4f}",
        "risk_score": calculate_fraud_score(probability),
        "risk_level": get_risk_level(probability),
        "transaction_features": transaction_features if isinstance(transaction_features, dict) else None
    }
    
    return report


def print_prediction_report(report):
    """
    Print a formatted prediction report
    
    Args:
        report: Report dictionary from create_prediction_report
    """
    print("\n" + "="*60)
    print("FRAUD DETECTION REPORT")
    print("="*60)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Model: {report['model_name']}")
    print(f"\nPrediction: {report['prediction']}")
    print(f"Fraud Probability: {report['fraud_probability']}")
    print(f"Risk Score: {report['risk_score']}/100")
    print(f"Risk Level: {report['risk_level']}")
    print("="*60 + "\n")


def batch_predict(model, transactions_df):
    """
    Predict fraud for multiple transactions
    
    Args:
        model: Trained model
        transactions_df: DataFrame of transactions
        
    Returns:
        DataFrame with predictions and probabilities
    """
    predictions = model.predict(transactions_df)
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(transactions_df)[:, 1]
    else:
        probabilities = predictions
    
    results_df = transactions_df.copy()
    results_df['Prediction'] = predictions
    results_df['Fraud_Probability'] = probabilities
    results_df['Risk_Score'] = (probabilities * 100).astype(int)
    results_df['Risk_Level'] = results_df['Fraud_Probability'].apply(get_risk_level)
    
    return results_df


def calculate_cost_benefit(confusion_matrix, cost_fp=1, cost_fn=100, benefit_tp=200):
    """
    Calculate cost-benefit analysis
    
    Args:
        confusion_matrix: 2x2 confusion matrix [[TN, FP], [FN, TP]]
        cost_fp: Cost of false positive (falsely flagging legitimate transaction)
        cost_fn: Cost of false negative (missing a fraud)
        benefit_tp: Benefit of catching a fraud
        
    Returns:
        Total cost/benefit and breakdown
    """
    tn, fp, fn, tp = confusion_matrix.ravel()
    
    total_cost_fp = fp * cost_fp
    total_cost_fn = fn * cost_fn
    total_benefit_tp = tp * benefit_tp
    
    net_benefit = total_benefit_tp - total_cost_fp - total_cost_fn
    
    analysis = {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "cost_false_positives": total_cost_fp,
        "cost_false_negatives": total_cost_fn,
        "benefit_true_positives": total_benefit_tp,
        "net_benefit": net_benefit
    }
    
    return analysis


def print_cost_benefit_analysis(analysis):
    """
    Print cost-benefit analysis
    
    Args:
        analysis: Analysis dictionary from calculate_cost_benefit
    """
    print("\n" + "="*60)
    print("COST-BENEFIT ANALYSIS")
    print("="*60)
    print(f"True Positives (Fraud Caught): {analysis['true_positives']:,}")
    print(f"False Positives (False Alarms): {analysis['false_positives']:,}")
    print(f"False Negatives (Missed Fraud): {analysis['false_negatives']:,}")
    print(f"True Negatives (Correct Legitimate): {analysis['true_negatives']:,}")
    print("\n" + "-"*60)
    print(f"Cost of False Positives: ${analysis['cost_false_positives']:,.2f}")
    print(f"Cost of False Negatives: ${analysis['cost_false_negatives']:,.2f}")
    print(f"Benefit from True Positives: ${analysis['benefit_true_positives']:,.2f}")
    print("-"*60)
    print(f"NET BENEFIT: ${analysis['net_benefit']:,.2f}")
    print("="*60 + "\n")


def create_deployment_package(model_path, feature_names, output_dir='deployment'):
    """
    Create a deployment package with model and metadata
    
    Args:
        model_path: Path to the trained model
        feature_names: List of feature names
        output_dir: Output directory for deployment package
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy model
    import shutil
    model_filename = os.path.basename(model_path)
    shutil.copy(model_path, os.path.join(output_dir, model_filename))
    
    # Create metadata
    metadata = {
        "model_file": model_filename,
        "feature_names": feature_names,
        "num_features": len(feature_names),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": "1.0.0"
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create prediction script
    prediction_script = """
import joblib
import pandas as pd
import json

# Load model and metadata
model = joblib.load('{}')
with open('metadata.json', 'r') as f:
    metadata = json.load(f)

def predict(transaction_data):
    '''
    Predict fraud for a transaction
    
    Args:
        transaction_data: dict with feature values
        
    Returns:
        dict with prediction and probability
    '''
    df = pd.DataFrame([transaction_data])
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0, 1]
    
    return {{
        'prediction': 'Fraud' if prediction == 1 else 'Legitimate',
        'probability': float(probability),
        'risk_score': int(probability * 100)
    }}

if __name__ == '__main__':
    # Example usage
    sample_transaction = {{}}  # Add sample features
    result = predict(sample_transaction)
    print(result)
""".format(model_filename)
    
    with open(os.path.join(output_dir, 'predict.py'), 'w') as f:
        f.write(prediction_script)
    
    print(f"Deployment package created in {output_dir}/")
    print(f"  - Model: {model_filename}")
    print(f"  - Metadata: metadata.json")
    print(f"  - Prediction script: predict.py")


def load_deployment_package(deployment_dir):
    """
    Load a deployment package
    
    Args:
        deployment_dir: Directory containing deployment package
        
    Returns:
        model and metadata
    """
    # Load metadata
    with open(os.path.join(deployment_dir, 'metadata.json'), 'r') as f:
        metadata = json.load(f)
    
    # Load model
    model_path = os.path.join(deployment_dir, metadata['model_file'])
    model = load_model(model_path)
    
    return model, metadata


# Main execution
if __name__ == "__main__":
    print("Utility module for Credit Card Fraud Detection")
    print("\nAvailable functions:")
    print("  - load_model()")
    print("  - save_model()")
    print("  - predict_transaction()")
    print("  - batch_predict()")
    print("  - calculate_cost_benefit()")
    print("  - create_deployment_package()")

