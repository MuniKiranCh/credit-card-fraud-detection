"""
Simplified Pipeline for Beginners
Easier to understand and follow
"""
import sys
import os
sys.path.append('src')

from src.data_preprocessing import DataPreprocessor
from src.model_training import ModelTrainer
import pandas as pd


def main():
    """Run the simplified pipeline"""
    
    print("\n" + "="*70)
    print(" CREDIT CARD FRAUD DETECTION - BEGINNER-FRIENDLY VERSION ")
    print("="*70)
    print("\nThis simplified version uses 3 models:")
    print("  1. Logistic Regression (Simple baseline)")
    print("  2. Random Forest (Tree-based)")
    print("  3. XGBoost (Advanced, production-ready)")
    print("\n" + "="*70)
    
    # Step 1: Check if data is preprocessed
    if not os.path.exists('data/processed/X_train.csv'):
        print("\n[STEP 1/2] Preprocessing Data...")
        print("-"*70)
        
        preprocessor = DataPreprocessor('data/creditcard.csv')
        
        try:
            preprocessor.run_full_pipeline()
        except FileNotFoundError:
            print("\n❌ ERROR: Dataset not found!")
            print("\nPlease download creditcard.csv from:")
            print("https://www.kaggle.com/mlg-ulb/creditcardfraud")
            print("\nPlace it in: data/creditcard.csv")
            return
    else:
        print("\n✓ Data already preprocessed")
    
    # Step 2: Load data
    print("\n[STEP 2/2] Training Models...")
    print("-"*70)
    
    X_train = pd.read_csv('data/processed/X_train.csv')
    X_test = pd.read_csv('data/processed/X_test.csv')
    y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
    
    # Step 3: Train models
    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    best_name, best_model = trainer.train_all()
    
    # Step 4: Save models
    trainer.save_models()
    
    # Final message
    print("\n" + "="*70)
    print(" SUCCESS! PROJECT COMPLETE! ")
    print("="*70)
    print("\n📚 WHAT TO DO NEXT:")
    print("\n1. UNDERSTAND YOUR RESULTS:")
    print("   • Open: models/model_comparison.csv")
    print(f"   • Best model: {best_name}")
    print("   • Memorize your accuracy, precision, recall scores!")
    
    print("\n2. TRY THE DEMO:")
    print("   • Run: python demo.py")
    print("   • See live fraud detection in action")
    
    print("\n3. LEARN THE CONCEPTS:")
    print("   • Read: LEARNING_GUIDE.md (starts here!)")
    print("   • This tells you exactly what to learn")
    
    print("\n4. EXPLORE NOTEBOOKS:")
    print("   • Open Jupyter: jupyter notebook")
    print("   • See visualizations and detailed analysis")
    
    print("\n5. PRACTICE EXPLANATION:")
    print("   • Can you explain SMOTE?")
    print("   • Can you explain why XGBoost is best?")
    print("   • Can you explain precision vs recall?")
    
    print("\n" + "="*70)
    print("💡 TIP: Focus on understanding, not memorizing!")
    print("   Start with LEARNING_GUIDE.md - it's made for beginners!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

