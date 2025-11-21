import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, mean_absolute_error, 
                             mean_squared_error, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

class CricketOutcomePredictionModel:
    """
    A comprehensive model to predict cricket shot outcomes (runs or wicket)
    based on delivery characteristics and shot type.
    """
    
    def __init__(self, csv_path):
        """
        Initialize the model with data from CSV file.
        
        Args:
            csv_path (str): Path to the cricket dataset CSV file
        """
        self.csv_path = csv_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        
    def load_and_explore_data(self):
        """Load data and perform exploratory data analysis."""
        print("=" * 80)
        print("LOADING AND EXPLORING DATA")
        print("=" * 80)
        
        # Load data
        self.data = pd.read_csv(self.csv_path)
        
        print(f"\nDataset Shape: {self.data.shape}")
        print(f"\nColumn Names:\n{self.data.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(self.data.head())
        
        print(f"\nData Types:")
        print(self.data.dtypes)
        
        print(f"\nMissing Values:")
        print(self.data.isnull().sum())
        
        print(f"\nBasic Statistics:")
        print(self.data.describe())
        
        # Check target variable distribution
        if 'Outcome' in self.data.columns:
            print(f"\nOutcome Distribution:")
            print(self.data['Outcome'].value_counts().sort_index())
        elif 'Runs' in self.data.columns or 'runs' in self.data.columns:
            runs_col = 'Runs' if 'Runs' in self.data.columns else 'runs'
            print(f"\nRuns Distribution:")
            print(self.data[runs_col].value_counts().sort_index())
        
        return self.data
    
    def preprocess_data(self, target_column='Outcome'):
        """
        Preprocess the data: encode categorical variables, normalize continuous variables.
        
        Args:
            target_column (str): Name of the target column (default: 'Outcome')
        """
        print("\n" + "=" * 80)
        print("PREPROCESSING DATA")
        print("=" * 80)
        
        # Make a copy to avoid modifying original
        df = self.data.copy()
        
        # Identify categorical and numerical columns
        # Expected features based on proposal
        expected_features = [
            'Bowler Type', 'BowlerType', 'bowler_type',
            'Ball Length', 'BallLength', 'ball_length', 'Length',
            'Ball Line', 'BallLine', 'ball_line', 'Line',
            'Speed', 'speed',
            'Shot Type', 'ShotType', 'shot_type', 'Shot',
            'Field Placement', 'FieldPlacement', 'field_placement', 'Field',
            'Angle', 'angle',
            'Bounce', 'bounce'
        ]
        
        # Find which columns actually exist (case-insensitive matching)
        df.columns = df.columns.str.strip()  # Remove whitespace
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower().replace('_', '').replace(' ', '')
            column_mapping[col] = col_lower
        
        # Identify categorical columns (excluding target)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)
        
        # Identify numerical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if target_column in numerical_cols:
            numerical_cols.remove(target_column)
        
        print(f"\nCategorical features: {categorical_cols}")
        print(f"Numerical features: {numerical_cols}")
        
        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            print(f"Encoded '{col}': {len(le.classes_)} unique values")
        
        # Prepare features and target
        feature_cols = categorical_cols + numerical_cols
        X = df[feature_cols]
        self.feature_names = feature_cols
        
        # Handle target variable
        if target_column not in df.columns:
            print(f"\nWarning: Target column '{target_column}' not found!")
            print(f"Available columns: {df.columns.tolist()}")
            # Try to find alternative
            possible_targets = ['Runs', 'runs', 'Wicket', 'wicket', 'outcome']
            for alt in possible_targets:
                if alt in df.columns:
                    print(f"Using '{alt}' as target instead")
                    target_column = alt
                    break
        
        y = df[target_column]
        
        # Normalize numerical features
        if numerical_cols:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
            print(f"\nNormalized {len(numerical_cols)} numerical features")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set size: {self.X_train.shape}")
        print(f"Testing set size: {self.X_test.shape}")
        print(f"\nTraining set outcome distribution:")
        print(pd.Series(self.y_train).value_counts().sort_index())
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, model_type='random_forest', **kwargs):
        """
        Train the prediction model.
        
        Args:
            model_type (str): Type of model ('random_forest' or 'gradient_boosting')
            **kwargs: Additional parameters for the model
        """
        print("\n" + "=" * 80)
        print(f"TRAINING {model_type.upper().replace('_', ' ')} MODEL")
        print("=" * 80)
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 20),
                min_samples_split=kwargs.get('min_samples_split', 5),
                min_samples_leaf=kwargs.get('min_samples_leaf', 2),
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 10),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=42,
                verbose=1
            )
        else:
            raise ValueError("model_type must be 'random_forest' or 'gradient_boosting'")
        
        print(f"\nModel parameters: {self.model.get_params()}")
        print("\nTraining in progress...")
        self.model.fit(self.X_train, self.y_train)
        print("✓ Training complete!")
        
        return self.model
    
    def evaluate_model(self, plot=True):
        """
        Comprehensive model evaluation with multiple metrics.
        
        Args:
            plot (bool): Whether to generate visualization plots
        """
        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Accuracy
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        print(f"\n{'Metric':<30} {'Training':<15} {'Testing':<15}")
        print("-" * 60)
        print(f"{'Accuracy':<30} {train_accuracy:<15.4f} {test_accuracy:<15.4f}")
        
        # F1 Score
        train_f1 = f1_score(self.y_train, y_train_pred, average='weighted')
        test_f1 = f1_score(self.y_test, y_test_pred, average='weighted')
        print(f"{'F1 Score (weighted)':<30} {train_f1:<15.4f} {test_f1:<15.4f}")
        
        # MAE and MSE (treating outcomes as ordinal)
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        
        print(f"{'Mean Absolute Error (MAE)':<30} {train_mae:<15.4f} {test_mae:<15.4f}")
        print(f"{'Mean Squared Error (MSE)':<30} {train_mse:<15.4f} {test_mse:<15.4f}")
        print(f"{'Root Mean Squared Error (RMSE)':<30} {np.sqrt(train_mse):<15.4f} {np.sqrt(test_mse):<15.4f}")
        
        # Detailed classification report
        print("\n" + "=" * 80)
        print("DETAILED CLASSIFICATION REPORT (Test Set)")
        print("=" * 80)
        print(classification_report(self.y_test, y_test_pred))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            print("\n" + "=" * 80)
            print("FEATURE IMPORTANCE")
            print("=" * 80)
            feature_importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': self.model.feature_importances_
            }).sort_values('Importance', ascending=False)
            print(feature_importance.to_string(index=False))
        
        if plot:
            self._plot_evaluation(y_test_pred, feature_importance if hasattr(self.model, 'feature_importances_') else None)
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'test_mae': test_mae,
            'test_mse': test_mse,
            'test_rmse': np.sqrt(test_mse)
        }
    
    def _plot_evaluation(self, y_test_pred, feature_importance):
        """Generate evaluation plots."""
        fig = plt.figure(figsize=(16, 10))
        
        # Confusion Matrix
        plt.subplot(2, 3, 1)
        cm = confusion_matrix(self.y_test, y_test_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')
        plt.ylabel('True Outcome')
        plt.xlabel('Predicted Outcome')
        
        # Outcome Distribution Comparison
        plt.subplot(2, 3, 2)
        outcome_comparison = pd.DataFrame({
            'Actual': self.y_test.value_counts().sort_index(),
            'Predicted': pd.Series(y_test_pred).value_counts().sort_index()
        })
        outcome_comparison.plot(kind='bar', ax=plt.gca())
        plt.title('Actual vs Predicted Distribution', fontsize=12, fontweight='bold')
        plt.xlabel('Outcome')
        plt.ylabel('Count')
        plt.legend()
        plt.xticks(rotation=0)
        
        # Feature Importance
        if feature_importance is not None:
            plt.subplot(2, 3, 3)
            top_features = feature_importance.head(10)
            plt.barh(range(len(top_features)), top_features['Importance'])
            plt.yticks(range(len(top_features)), top_features['Feature'])
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importance', fontsize=12, fontweight='bold')
            plt.gca().invert_yaxis()
        
        # Prediction Errors
        plt.subplot(2, 3, 4)
        errors = np.abs(self.y_test - y_test_pred)
        plt.hist(errors, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Absolute Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
        
        # Actual vs Predicted Scatter
        plt.subplot(2, 3, 5)
        plt.scatter(self.y_test, y_test_pred, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Outcome')
        plt.ylabel('Predicted Outcome')
        plt.title('Actual vs Predicted Outcomes', fontsize=12, fontweight='bold')
        plt.legend()
        
        # Class-wise Accuracy
        plt.subplot(2, 3, 6)
        class_accuracy = []
        classes = sorted(self.y_test.unique())
        for cls in classes:
            mask = self.y_test == cls
            if mask.sum() > 0:
                acc = accuracy_score(self.y_test[mask], y_test_pred[mask])
                class_accuracy.append(acc)
            else:
                class_accuracy.append(0)
        
        plt.bar(range(len(classes)), class_accuracy)
        plt.xticks(range(len(classes)), classes)
        plt.ylabel('Accuracy')
        plt.xlabel('Outcome Class')
        plt.title('Per-Class Accuracy', fontsize=12, fontweight='bold')
        plt.ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        print("\n✓ Evaluation plots saved as 'model_evaluation.png'")
        plt.show()
    
    def predict_probability(self, delivery_features):
        """
        Predict probability distribution for a given delivery.
        
        Args:
            delivery_features (dict or pd.DataFrame): Delivery characteristics
            
        Returns:
            dict: Probability distribution over outcomes
        """
        if isinstance(delivery_features, dict):
            delivery_features = pd.DataFrame([delivery_features])
        
        # Encode and normalize
        for col in self.label_encoders.keys():
            if col in delivery_features.columns:
                delivery_features[col] = self.label_encoders[col].transform(
                    delivery_features[col].astype(str)
                )
        
        probabilities = self.model.predict_proba(delivery_features[self.feature_names])
        classes = self.model.classes_
        
        return dict(zip(classes, probabilities[0]))


# Example usage
if __name__ == "__main__":
    # Initialize model with your CSV file
    model = CricketOutcomePredictionModel('cricket_shot_selection_updated.csv')
    
    # Step 1: Load and explore data
    data = model.load_and_explore_data()
    
    # Step 2: Preprocess data
    # Adjust target_column name based on your CSV (e.g., 'Outcome', 'Runs', etc.)
    X_train, X_test, y_train, y_test = model.preprocess_data(target_column='Outcome')
    
    # Step 3: Train model
    # Try Random Forest first (as per proposal)
    trained_model = model.train_model(model_type='random_forest', n_estimators=200)
    
    # Step 4: Evaluate model
    metrics = model.evaluate_model(plot=True)
    
    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nFinal Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Final Test MAE: {metrics['test_mae']:.4f}")
    print(f"\nModel ready for RL environment simulation!")
    
    # Optional: Try Gradient Boosting for comparison
    # model.train_model(model_type='gradient_boosting', n_estimators=200)
    # model.evaluate_model(plot=True)