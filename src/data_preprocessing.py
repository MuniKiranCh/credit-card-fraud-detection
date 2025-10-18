"""
Data Preprocessing Module for Credit Card Fraud Detection
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split
import os


class DataPreprocessor:
    """
    A comprehensive data preprocessing class for credit card fraud detection
    """
    
    def __init__(self, data_path='data/creditcard.csv'):
        """
        Initialize the preprocessor
        
        Args:
            data_path: Path to the raw dataset
        """
        self.data_path = data_path
        self.scaler = RobustScaler()
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load the raw dataset"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Please download from https://www.kaggle.com/mlg-ulb/creditcardfraud"
            )
        
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded: {self.df.shape[0]:,} rows x {self.df.shape[1]} columns")
        return self.df
    
    def check_data_quality(self):
        """Check for missing values and duplicates"""
        missing = self.df.isnull().sum().sum()
        duplicates = self.df.duplicated().sum()
        
        print("\nData Quality Check:")
        print(f"Missing values: {missing}")
        print(f"Duplicate rows: {duplicates}")
        
        return missing, duplicates
    
    def create_features(self):
        """Create additional features"""
        df = self.df.copy()
        
        # Time-based features
        df['Time_Hour'] = (df['Time'] / 3600) % 24
        df['Time_Day'] = (df['Time'] / 86400).astype(int)
        
        # Cyclical encoding for time
        df['Time_Hour_Sin'] = np.sin(2 * np.pi * df['Time_Hour'] / 24)
        df['Time_Hour_Cos'] = np.cos(2 * np.pi * df['Time_Hour'] / 24)
        
        # Amount features
        df['Amount_log'] = np.log1p(df['Amount'])
        
        # Scale Amount
        df['Amount_scaled'] = self.scaler.fit_transform(df[['Amount']])
        
        self.df = df
        print("\nFeatures created:")
        print("- Time_Hour, Time_Day")
        print("- Time_Hour_Sin, Time_Hour_Cos (cyclical encoding)")
        print("- Amount_log, Amount_scaled")
        
        return df
    
    def get_features_and_target(self):
        """Extract features and target variable"""
        # Get PCA features (V1-V28)
        pca_features = [col for col in self.df.columns if col.startswith('V')]
        
        # Additional engineered features
        additional_features = [
            'Amount_scaled', 'Time_Hour_Sin', 
            'Time_Hour_Cos', 'Amount_log'
        ]
        
        feature_columns = pca_features + additional_features
        
        X = self.df[feature_columns]
        y = self.df['Class']
        
        return X, y, feature_columns
    
    def split_data(self, test_size=0.2, random_state=42):
        """
        Split data into train and test sets with stratification
        
        Args:
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
        """
        X, y, feature_columns = self.get_features_and_target()
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        
        print(f"\nTrain-Test Split:")
        print(f"Training set: {self.X_train.shape[0]:,} samples")
        print(f"Test set: {self.X_test.shape[0]:,} samples")
        print(f"\nFraud rate in train: {self.y_train.sum()/len(self.y_train)*100:.3f}%")
        print(f"Fraud rate in test: {self.y_test.sum()/len(self.y_test)*100:.3f}%")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_processed_data(self, output_dir='data/processed'):
        """
        Save processed data to CSV files
        
        Args:
            output_dir: Directory to save processed data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save processed full dataset
        self.df.to_csv(f'{output_dir}/creditcard_processed.csv', index=False)
        
        # Save train-test splits
        self.X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
        self.X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
        pd.DataFrame(self.y_train, columns=['Class']).to_csv(
            f'{output_dir}/y_train.csv', index=False
        )
        pd.DataFrame(self.y_test, columns=['Class']).to_csv(
            f'{output_dir}/y_test.csv', index=False
        )
        
        # Save feature names
        with open(f'{output_dir}/feature_names.txt', 'w') as f:
            f.write('\n'.join(self.X_train.columns))
        
        print(f"\nProcessed data saved to {output_dir}/")
    
    def run_full_pipeline(self):
        """Run the complete preprocessing pipeline"""
        print("="*60)
        print("Starting Data Preprocessing Pipeline")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Check quality
        self.check_data_quality()
        
        # Create features
        self.create_features()
        
        # Split data
        self.split_data()
        
        # Save data
        self.save_processed_data()
        
        print("\n" + "="*60)
        print("Preprocessing Complete!")
        print("="*60)
        
        return self.X_train, self.X_test, self.y_train, self.y_test


def main():
    """Main execution function"""
    preprocessor = DataPreprocessor('../data/creditcard.csv')
    X_train, X_test, y_train, y_test = preprocessor.run_full_pipeline()
    
    return preprocessor


if __name__ == "__main__":
    main()

