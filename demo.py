"""
Quick Demo Script - Test a single prediction
"""
import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils import (
    load_model, 
    predict_transaction,
    create_prediction_report,
    print_prediction_report,
    get_risk_level
)


def demo_prediction():
    """Demo a single fraud prediction"""
    
    print("\n" + "="*70)
    print("CREDIT CARD FRAUD DETECTION - DEMO")
    print("="*70)
    
    # Check if model exists
    model_path = 'models/xgboost.pkl'
    if not os.path.exists(model_path):
        print("\nModel not found! Please run the training pipeline first:")
        print("  python run_pipeline.py")
        return
    
    # Load model
    print("\nLoading trained model...")
    model = load_model(model_path)
    print(f"Model loaded: {model_path}")
    
    # Load test data
    print("\nLoading test data...")
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    # Select random transactions
    print("\nTesting on 5 random transactions...")
    print("-"*70)
    
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    
    correct = 0
    for i, idx in enumerate(sample_indices, 1):
        transaction = X_test.iloc[idx:idx+1]
        actual_label = y_test[idx]
        
        # Predict
        prediction, probability = predict_transaction(model, transaction)
        
        # Check if correct
        is_correct = (prediction == actual_label)
        correct += is_correct
        
        print(f"\nTransaction {i}:")
        print(f"  Predicted: {'Fraud' if prediction == 1 else 'Legitimate'}")
        print(f"  Actual: {'Fraud' if actual_label == 1 else 'Legitimate'}")
        print(f"  Fraud Probability: {probability:.4f}")
        print(f"  Risk Level: {get_risk_level(probability)}")
        print(f"  Correct: {'✓ Yes' if is_correct else '✗ No'}")
    
    print("\n" + "-"*70)
    print(f"Accuracy on sample: {correct}/5 ({correct/5*100:.0f}%)")
    print("="*70 + "\n")


def interactive_demo():
    """Interactive demo with user input"""
    
    print("\n" + "="*70)
    print("INTERACTIVE FRAUD DETECTION DEMO")
    print("="*70)
    
    # Load model
    model_path = 'models/xgboost.pkl'
    if not os.path.exists(model_path):
        print("\nModel not found! Please run the training pipeline first.")
        return
    
    model = load_model(model_path)
    
    # Load feature names
    with open('data/processed/feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f]
    
    print(f"\nModel loaded: {model_path}")
    print(f"Number of features required: {len(feature_names)}")
    
    print("\nNote: This model requires 32 features (V1-V28, Amount_scaled, etc.)")
    print("For simplicity, we'll load from test data.")
    
    # Load test data
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    while True:
        print("\n" + "-"*70)
        choice = input("\nEnter transaction index (0-{}) or 'q' to quit: ".format(len(X_test)-1))
        
        if choice.lower() == 'q':
            print("Exiting demo...")
            break
        
        try:
            idx = int(choice)
            if idx < 0 or idx >= len(X_test):
                print(f"Invalid index! Please enter 0-{len(X_test)-1}")
                continue
            
            # Get transaction
            transaction = X_test.iloc[idx:idx+1]
            actual_label = y_test[idx]
            
            # Predict
            prediction, probability = predict_transaction(model, transaction)
            
            # Create report
            report = create_prediction_report(
                transaction_features={},  # Don't show all 32 features
                prediction=prediction,
                probability=probability,
                model_name="XGBoost"
            )
            
            print_prediction_report(report)
            print(f"Actual Label: {'Fraud' if actual_label == 1 else 'Legitimate'}")
            
            if prediction == actual_label:
                print("✓ Prediction is CORRECT!")
            else:
                print("✗ Prediction is INCORRECT!")
                
        except ValueError:
            print("Invalid input! Please enter a number or 'q'")
    
    print("\nThank you for using the demo!")
    print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_demo()
    else:
        demo_prediction()

