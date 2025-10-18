"""
Model Training Module
Focus on 3 key models: Logistic Regression, Random Forest, XGBoost
"""
import numpy as np
import pandas as pd
import time
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, matthews_corrcoef,
    classification_report, confusion_matrix
)

from imblearn.over_sampling import SMOTE
import xgboost as xgb


class ModelTrainer:
    """
    Model trainer with 3 essential algorithms
    """
    
    def __init__(self, X_train, X_test, y_train, y_test):
        """Initialize with train/test data"""
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Apply SMOTE to balance the data
        print("\n" + "="*60)
        print("Applying SMOTE to balance training data...")
        print("="*60)
        smote = SMOTE(random_state=42, n_jobs=-1)
        self.X_train_balanced, self.y_train_balanced = smote.fit_resample(
            self.X_train, self.y_train
        )
        print(f"✓ Original samples: {len(self.X_train):,}")
        print(f"✓ After SMOTE: {len(self.X_train_balanced):,}")
        print(f"✓ Fraud rate: 50% (balanced)\n")
        
        self.models = {}
        self.results = []
    
    def evaluate_model(self, model, model_name):
        """Evaluate a trained model and return metrics"""
        print(f"\n{'='*60}")
        print(f"Evaluating {model_name}...")
        print(f"{'='*60}")
        
        # Make predictions
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        # Calculate all metrics
        metrics = {
            'Model': model_name,
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred),
            'Recall': recall_score(self.y_test, y_pred),
            'F1-Score': f1_score(self.y_test, y_pred),
            'ROC-AUC': roc_auc_score(self.y_test, y_pred_proba)
        }
        
        # Print results
        print(f"\n✓ Accuracy:  {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)")
        print(f"✓ Precision: {metrics['Precision']:.4f} (Of fraud predictions, {metrics['Precision']*100:.0f}% correct)")
        print(f"✓ Recall:    {metrics['Recall']:.4f} (Caught {metrics['Recall']*100:.0f}% of actual frauds)")
        print(f"✓ F1-Score:  {metrics['F1-Score']:.4f}")
        print(f"✓ ROC-AUC:   {metrics['ROC-AUC']:.4f}")
        
        # Show confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {tn:,} (Correctly identified legitimate)")
        print(f"  False Positives: {fp:,} (False alarms)")
        print(f"  False Negatives: {fn:,} (Missed frauds) ⚠️")
        print(f"  True Positives:  {tp:,} (Caught frauds) ✓")
        
        return metrics
    
    def train_logistic_regression(self):
        """
        Model 1: Logistic Regression (Simple Baseline)
        - Fast and simple
        - Good for understanding feature relationships
        - Linear model
        """
        print("\n" + "="*60)
        print("MODEL 1: Logistic Regression (Baseline)")
        print("="*60)
        print("ℹ️  Simple linear model - fast and interpretable")
        
        model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        
        print("\nTraining...")
        start_time = time.time()
        model.fit(self.X_train_balanced, self.y_train_balanced)
        train_time = time.time() - start_time
        print(f"✓ Training completed in {train_time:.2f} seconds")
        
        metrics = self.evaluate_model(model, 'Logistic Regression')
        self.models['logistic_regression'] = model
        self.results.append(metrics)
        
        return model
    
    def train_random_forest(self):
        """
        Model 2: Random Forest (Tree-Based Ensemble)
        - Uses multiple decision trees
        - Handles non-linear patterns
        - Shows feature importance
        """
        print("\n" + "="*60)
        print("MODEL 2: Random Forest (Tree Ensemble)")
        print("="*60)
        print("ℹ️  Uses 100 decision trees - better for non-linear patterns")
        
        model = RandomForestClassifier(
            n_estimators=100,      # Number of trees
            max_depth=20,          # Maximum depth of each tree
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        print("\nTraining 100 trees...")
        start_time = time.time()
        model.fit(self.X_train_balanced, self.y_train_balanced)
        train_time = time.time() - start_time
        print(f"✓ Training completed in {train_time:.2f} seconds")
        
        metrics = self.evaluate_model(model, 'Random Forest')
        self.models['random_forest'] = model
        self.results.append(metrics)
        
        # Show top features
        feature_importance = pd.DataFrame({
            'feature': self.X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 5 Most Important Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        return model
    
    def train_xgboost(self):
        """
        Model 3: XGBoost (Advanced Gradient Boosting)
        - Industry-standard algorithm
        - Usually best performance
        - Used by Kaggle winners
        """
        print("\n" + "="*60)
        print("MODEL 3: XGBoost (Advanced - Production Ready)")
        print("="*60)
        print("ℹ️  State-of-the-art gradient boosting - industry standard")
        
        model = xgb.XGBClassifier(
            n_estimators=100,      # Number of boosting rounds
            max_depth=6,           # Depth of trees
            learning_rate=0.1,     # How fast it learns
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        print("\nTraining with gradient boosting...")
        start_time = time.time()
        model.fit(self.X_train_balanced, self.y_train_balanced)
        train_time = time.time() - start_time
        print(f"✓ Training completed in {train_time:.2f} seconds")
        
        metrics = self.evaluate_model(model, 'XGBoost')
        self.models['xgboost'] = model
        self.results.append(metrics)
        
        return model
    
    def train_all(self):
        """Train all 3 models and compare"""
        print("\n" + "="*70)
        print("TRAINING ALL MODELS")
        print("="*70)
        
        # Train each model
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        
        # Show comparison
        self.show_comparison()
        
        return self.get_best_model()
    
    def show_comparison(self):
        """Show comparison of all models"""
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        results_df = pd.DataFrame(self.results)
        print("\n" + results_df.to_string(index=False))
        
        # Find best model
        best_idx = results_df['ROC-AUC'].idxmax()
        best_model = results_df.loc[best_idx, 'Model']
        best_score = results_df.loc[best_idx, 'ROC-AUC']
        
        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"   ROC-AUC Score: {best_score:.4f}")
        
        # Show which model is best for what
        print(f"\n💡 Model Recommendations:")
        print(f"   • Fastest: Logistic Regression")
        print(f"   • Most Interpretable: Random Forest (feature importance)")
        print(f"   • Best Performance: {best_model} ⭐")
        print(f"   • Production Ready: XGBoost")
    
    def get_best_model(self):
        """Return the best performing model"""
        results_df = pd.DataFrame(self.results)
        best_idx = results_df['ROC-AUC'].idxmax()
        best_model_name = results_df.loc[best_idx, 'Model']
        
        model_map = {
            'Logistic Regression': 'logistic_regression',
            'Random Forest': 'random_forest',
            'XGBoost': 'xgboost'
        }
        
        best_model_key = model_map[best_model_name]
        return best_model_name, self.models[best_model_key]
    
    def save_models(self, output_dir='models'):
        """Save all trained models"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print("SAVING MODELS")
        print(f"{'='*60}")
        
        for name, model in self.models.items():
            filepath = f'{output_dir}/{name}.pkl'
            joblib.dump(model, filepath)
            print(f"✓ Saved: {filepath}")
        
        # Save comparison results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv(f'{output_dir}/model_comparison.csv', index=False)
        print(f"✓ Saved: {output_dir}/model_comparison.csv")
        
        print(f"\n✓ All models saved successfully!")


def main():
    """Main function to run training"""
    print("\n" + "="*70)
    print("CREDIT CARD FRAUD DETECTION - SIMPLIFIED")
    print("Train 3 models: Logistic Regression, Random Forest, XGBoost")
    print("="*70)
    
    # Load preprocessed data
    print("\nLoading data...")
    X_train = pd.read_csv('../data/processed/X_train.csv')
    X_test = pd.read_csv('../data/processed/X_test.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv').values.ravel()
    y_test = pd.read_csv('../data/processed/y_test.csv').values.ravel()
    
    print(f"✓ Training set: {X_train.shape[0]:,} samples")
    print(f"✓ Test set: {X_test.shape[0]:,} samples")
    print(f"✓ Features: {X_train.shape[1]}")
    print(f"✓ Fraud rate in test: {y_test.sum()/len(y_test)*100:.3f}%")
    
    # Train all models
    trainer = ModelTrainer(X_train, X_test, y_train, y_test)
    best_name, best_model = trainer.train_all()
    
    # Save models
    trainer.save_models()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE! ✓")
    print("="*70)
    print(f"\n🎯 Next Steps:")
    print(f"   1. Run demo: python demo.py")
    print(f"   2. Check saved models in: models/")
    print(f"   3. Open Jupyter notebooks for detailed analysis")
    print(f"\n💡 Tip: Study LEARNING_GUIDE.md to understand everything!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

