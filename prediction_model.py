import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                             r2_score, explained_variance_score)
import matplotlib.pyplot as plt
import seaborn as sns

class CricketRewardPredictionModel:
    """
    A model to predict the reward (runs scored or wicket penalty) 
    given delivery conditions (state) and shot type (action).
    
    This model learns: R(state, action) → reward
    where reward = -100 for wicket, or {0, 1, 2, 3, 4, 6} for runs scored
    """
    
    def __init__(self, csv_path, wicket_penalty=-100):
        """
        Initialize the reward prediction model.
        
        Args:
            csv_path (str): Path to the cricket dataset CSV file
            wicket_penalty (int): Negative reward for getting out (default: -100)
        """
        self.csv_path = csv_path
        self.wicket_penalty = wicket_penalty
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
        
        return self.data
    
    def preprocess_data(self, runs_column='Runs', wicket_column='Wicket'):
        """
        Preprocess the data and create the reward target variable.
        
        STATE features (delivery conditions):
        - Bowler Type, Ball Length, Ball Line, Speed, Field Placement, Angle, Bounce
        
        ACTION feature:
        - Shot Type
        
        TARGET (reward):
        - Runs scored if not out, or wicket_penalty if out
        
        Args:
            runs_column (str): Name of the runs column
            wicket_column (str): Name of the wicket column (1 = out, 0 = not out)
        """
        print("\n" + "=" * 80)
        print("PREPROCESSING DATA")
        print("=" * 80)
        
        df = self.data.copy()
        df.columns = df.columns.str.strip()
        
        # Find the actual column names (case-insensitive)
        column_map = {col.lower().replace('_', '').replace(' ', ''): col 
                      for col in df.columns}
        
        # Locate runs and wicket columns
        runs_col = None
        wicket_col = None
        
        for possible_name in ['runs', 'run', 'runsscored']:
            if possible_name in column_map:
                runs_col = column_map[possible_name]
                break
        
        for possible_name in ['wicket', 'out', 'dismissed', 'iswwicket']:
            if possible_name in column_map:
                wicket_col = column_map[possible_name]
                break
        
        if runs_col is None:
            print(f"Warning: Could not find runs column. Available columns: {df.columns.tolist()}")
            print("Please specify the correct column name.")
            return None
        
        print(f"\nUsing '{runs_col}' as runs column")
        if wicket_col:
            print(f"Using '{wicket_col}' as wicket column")
        
        # CREATE REWARD TARGET
        # Reward = wicket_penalty if wicket, else runs scored
        if wicket_col and wicket_col in df.columns:
            df['Reward'] = df.apply(
                lambda row: self.wicket_penalty if row[wicket_col] == 1 else row[runs_col],
                axis=1
            )
            print(f"\nCreated Reward column: {self.wicket_penalty} for wickets, runs otherwise")
        else:
            # If no wicket column, assume all rows are valid deliveries (no wickets in data)
            df['Reward'] = df[runs_col]
            print(f"\nNo wicket column found. Using runs as reward directly.")
        
        print(f"\nReward Distribution:")
        print(df['Reward'].value_counts().sort_index())
        print(f"\nReward Statistics:")
        print(df['Reward'].describe())
        
        # Identify STATE features (delivery conditions) and ACTION feature (shot type)
        # Expected STATE features
        state_features = []
        action_feature = None
        
        # Map common column name variations
        feature_mapping = {
            'bowlertype': ['bowlertype', 'bowler_type', 'bowler type', 'bowler'],
            'length': ['length', 'ball_length', 'ball length', 'balllength'],
            'line': ['line', 'ball_line', 'ball line', 'ballline'],
            'speed': ['speed', 'ball_speed', 'ball speed', 'velocity'],
            'shottype': ['shottype', 'shot_type', 'shot type', 'shot'],
            'fieldplacement': ['fieldplacement', 'field_placement', 'field placement', 'field'],
            'angle': ['angle', 'ball_angle', 'ball angle'],
            'bounce': ['bounce', 'ball_bounce', 'ball bounce']
        }
        
        actual_columns = {}
        for feature, variations in feature_mapping.items():
            for var in variations:
                if var in column_map:
                    actual_columns[feature] = column_map[var]
                    break
        
        print(f"\nIdentified columns:")
        for feature, col in actual_columns.items():
            print(f"  {feature}: '{col}'")
        
        # Separate state and action features
        action_feature = actual_columns.get('shottype')
        state_features = [col for key, col in actual_columns.items() if key != 'shottype']
        
        if action_feature is None:
            print("\nWarning: Shot Type column not found! This is required for the model.")
            print("The model needs Shot Type as the ACTION in R(state, action)")
        
        all_features = state_features + ([action_feature] if action_feature else [])
        
        # Identify categorical and numerical columns
        categorical_cols = []
        numerical_cols = []
        
        for col in all_features:
            if df[col].dtype == 'object':
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        
        print(f"\nSTATE features (delivery conditions): {state_features}")
        print(f"ACTION feature (shot type): {action_feature}")
        print(f"\nCategorical features: {categorical_cols}")
        print(f"Numerical features: {numerical_cols}")
        
        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            print(f"Encoded '{col}': {len(le.classes_)} unique values")
        
        # Prepare features (X) and target (y)
        X = df[all_features].copy()
        y = df['Reward'].copy()
        self.feature_names = all_features
        
        # Normalize numerical features
        if numerical_cols:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
            print(f"\nNormalized {len(numerical_cols)} numerical features")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"\nTraining set size: {self.X_train.shape}")
        print(f"Testing set size: {self.X_test.shape}")
        print(f"\nTraining set reward statistics:")
        print(self.y_train.describe())
        
        # Show distribution of rewards
        print(f"\nTraining reward distribution:")
        unique, counts = np.unique(self.y_train, return_counts=True)
        for val, count in zip(unique, counts):
            print(f"  Reward {val}: {count} ({count/len(self.y_train)*100:.1f}%)")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, model_type='random_forest', **kwargs):
        """
        Train the reward prediction model using regression.
        
        Args:
            model_type (str): Type of model ('random_forest' or 'gradient_boosting')
            **kwargs: Additional parameters for the model
        """
        print("\n" + "=" * 80)
        print(f"TRAINING {model_type.upper().replace('_', ' ')} REGRESSOR")
        print("=" * 80)
        print("\nThis model learns: R(state, action) → reward")
        print(f"Where reward = {self.wicket_penalty} for wicket, or runs scored (0-6)")
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 200),
                max_depth=kwargs.get('max_depth', 20),
                min_samples_split=kwargs.get('min_samples_split', 5),
                min_samples_leaf=kwargs.get('min_samples_leaf', 2),
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
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
        print("Training complete!")
        
        return self.model
    
    def evaluate_model(self, plot=True):
        """
        Comprehensive model evaluation for reward prediction.
        
        Args:
            plot (bool): Whether to generate visualization plots
        """
        print("\n" + "=" * 80)
        print("MODEL EVALUATION - REWARD PREDICTION")
        print("=" * 80)
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Regression metrics
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        
        train_mse = mean_squared_error(self.y_train, y_train_pred)
        test_mse = mean_squared_error(self.y_test, y_test_pred)
        
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        train_evs = explained_variance_score(self.y_train, y_train_pred)
        test_evs = explained_variance_score(self.y_test, y_test_pred)
        
        print(f"\n{'Metric':<35} {'Training':<15} {'Testing':<15}")
        print("-" * 65)
        print(f"{'Mean Absolute Error (MAE)':<35} {train_mae:<15.4f} {test_mae:<15.4f}")
        print(f"{'Root Mean Squared Error (RMSE)':<35} {train_rmse:<15.4f} {test_rmse:<15.4f}")
        print(f"{'R² Score':<35} {train_r2:<15.4f} {test_r2:<15.4f}")
        print(f"{'Explained Variance Score':<35} {train_evs:<15.4f} {test_evs:<15.4f}")
        
        # Prediction accuracy for discrete rewards
        print("\n" + "=" * 80)
        print("DISCRETE REWARD PREDICTION ACCURACY")
        print("=" * 80)
        
        # Round predictions to nearest valid reward
        valid_rewards = sorted(self.y_test.unique())
        y_test_rounded = self._round_to_valid_rewards(y_test_pred, valid_rewards)
        
        accuracy = np.mean(y_test_rounded == self.y_test)
        print(f"\nExact Match Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Calculate accuracy for wickets vs runs separately
        wicket_mask_test = self.y_test == self.wicket_penalty
        runs_mask_test = self.y_test != self.wicket_penalty
        
        if wicket_mask_test.sum() > 0:
            wicket_accuracy = np.mean(y_test_rounded[wicket_mask_test] == self.y_test[wicket_mask_test])
            print(f"Wicket Prediction Accuracy: {wicket_accuracy:.4f} ({wicket_accuracy*100:.2f}%)")
        
        if runs_mask_test.sum() > 0:
            runs_accuracy = np.mean(y_test_rounded[runs_mask_test] == self.y_test[runs_mask_test])
            print(f"Runs Prediction Accuracy: {runs_accuracy:.4f} ({runs_accuracy*100:.2f}%)")
        
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
            self._plot_evaluation(y_test_pred, y_test_rounded, 
                                 feature_importance if hasattr(self.model, 'feature_importances_') else None)
        
        return {
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_accuracy': accuracy,
            'feature_importance': feature_importance if hasattr(self.model, 'feature_importances_') else None
        }
    
    def _round_to_valid_rewards(self, predictions, valid_rewards):
        """Round continuous predictions to nearest valid discrete reward."""
        rounded = np.zeros_like(predictions)
        for i, pred in enumerate(predictions):
            rounded[i] = min(valid_rewards, key=lambda x: abs(x - pred))
        return rounded
    
    def _plot_evaluation(self, y_test_pred, y_test_rounded, feature_importance):
        """Generate evaluation plots for reward prediction."""
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Actual vs Predicted (Continuous)
        plt.subplot(3, 3, 1)
        plt.scatter(self.y_test, y_test_pred, alpha=0.5, s=20)
        min_val = min(self.y_test.min(), y_test_pred.min())
        max_val = max(self.y_test.max(), y_test_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Reward')
        plt.ylabel('Predicted Reward (Continuous)')
        plt.title('Actual vs Predicted Rewards', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Actual vs Predicted (Rounded)
        plt.subplot(3, 3, 2)
        plt.scatter(self.y_test, y_test_rounded, alpha=0.5, s=20)
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        plt.xlabel('Actual Reward')
        plt.ylabel('Predicted Reward (Rounded)')
        plt.title('Actual vs Predicted (Discrete)', fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Prediction Error Distribution
        plt.subplot(3, 3, 3)
        errors = self.y_test - y_test_pred
        plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('Prediction Error (Actual - Predicted)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors', fontsize=12, fontweight='bold')
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
        plt.grid(True, alpha=0.3)
        
        # 4. Reward Distribution Comparison
        plt.subplot(3, 3, 4)
        # Get all unique rewards from both actual and predicted
        all_unique_rewards = sorted(set(self.y_test.unique()) | set(y_test_rounded))
        
        actual_counts = [np.sum(self.y_test == val) for val in all_unique_rewards]
        predicted_counts = [np.sum(y_test_rounded == val) for val in all_unique_rewards]
        
        x = np.arange(len(all_unique_rewards))
        width = 0.35
        plt.bar(x - width/2, actual_counts, width, label='Actual', alpha=0.8)
        plt.bar(x + width/2, predicted_counts, width, label='Predicted', alpha=0.8)
        plt.xlabel('Reward Value')
        plt.ylabel('Count')
        plt.title('Actual vs Predicted Distribution', fontsize=12, fontweight='bold')
        plt.xticks(x, all_unique_rewards)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # 5. Feature Importance
        if feature_importance is not None:
            plt.subplot(3, 3, 5)
            top_features = feature_importance.head(min(10, len(feature_importance)))
            plt.barh(range(len(top_features)), top_features['Importance'])
            plt.yticks(range(len(top_features)), top_features['Feature'])
            plt.xlabel('Importance')
            plt.title('Feature Importance', fontsize=12, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
        
        # 6. Residuals Plot
        plt.subplot(3, 3, 6)
        residuals = self.y_test - y_test_pred
        plt.scatter(y_test_pred, residuals, alpha=0.5, s=20)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Reward')
        plt.ylabel('Residual (Actual - Predicted)')
        plt.title('Residual Plot', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 7. Per-Reward Accuracy
        plt.subplot(3, 3, 7)
        unique_rewards = sorted(self.y_test.unique())
        accuracies = []
        for reward in unique_rewards:
            mask = self.y_test == reward
            if mask.sum() > 0:
                acc = np.mean(y_test_rounded[mask] == self.y_test[mask])
                accuracies.append(acc)
            else:
                accuracies.append(0)
        
        plt.bar(range(len(unique_rewards)), accuracies)
        plt.xticks(range(len(unique_rewards)), unique_rewards)
        plt.ylabel('Accuracy')
        plt.xlabel('Reward Value')
        plt.title('Per-Reward Prediction Accuracy', fontsize=12, fontweight='bold')
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3, axis='y')
        
        # 8. Cumulative Error
        plt.subplot(3, 3, 8)
        sorted_errors = np.sort(np.abs(errors))
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        plt.plot(sorted_errors, cumulative, linewidth=2)
        plt.xlabel('Absolute Error')
        plt.ylabel('Cumulative Proportion')
        plt.title('Cumulative Error Distribution', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # 9. Error by Reward Type
        plt.subplot(3, 3, 9)
        reward_types = sorted(self.y_test.unique())
        errors_by_reward = [errors[self.y_test == r] for r in reward_types]
        plt.boxplot(errors_by_reward, labels=reward_types)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=1)
        plt.xlabel('Actual Reward')
        plt.ylabel('Prediction Error')
        plt.title('Error Distribution by Reward Type', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('reward_model_evaluation.png', dpi=300, bbox_inches='tight')
        print("\n✓ Evaluation plots saved as 'reward_model_evaluation.png'")
        plt.show()
    
    def predict_reward(self, state, action):
        """
        Predict the expected reward for a given state-action pair.
        
        Args:
            state (dict): Delivery conditions (Bowler Type, Line, Length, Speed, etc.)
            action (str): Shot type
            
        Returns:
            float: Expected reward (runs or wicket penalty)
        """
        # Combine state and action
        features = {**state, **{'Shot Type': action}}
        df = pd.DataFrame([features])
        
        # Encode categorical features
        for col in self.label_encoders.keys():
            if col in df.columns:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Normalize numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numerical_cols:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        # Predict
        reward = self.model.predict(df[self.feature_names])[0]
        return reward


# Example usage for RL integration
if __name__ == "__main__":
    # Initialize model with your CSV file
    model = CricketRewardPredictionModel('cricket_shot_selection_updated.csv', wicket_penalty=-100)
    
    # Step 1: Load and explore data
    print("\nSTEP 1: DATA LOADING")
    data = model.load_and_explore_data()
    
    # Step 2: Preprocess data
    # Adjust column names based on your actual CSV
    print("\nSTEP 2: DATA PREPROCESSING")
    X_train, X_test, y_train, y_test = model.preprocess_data(
        runs_column='Runs',  # Adjust to your column name
        wicket_column='Wicket'  # Adjust to your column name
    )
    
    # Step 3: Train model
    print("\nSTEP 3: MODEL TRAINING")
    trained_model = model.train_model(model_type='random_forest', n_estimators=200)
    
    # Step 4: Evaluate model
    print("\nSTEP 4: MODEL EVALUATION")
    metrics = model.evaluate_model(plot=True)
    
    print("\n" + "=" * 80)
    print("REWARD PREDICTION MODEL COMPLETE!")
    print("=" * 80)
    print(f"\nTest MAE: {metrics['test_mae']:.4f}")
    print(f"Test RMSE: {metrics['test_rmse']:.4f}")
    print(f"Test R²: {metrics['test_r2']:.4f}")
    print(f"Discrete Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"\nModel ready for RL environment!")
    print(f"   Use model.predict_reward(state, action) to get rewards")
    
    # Example prediction for RL
    print("\n" + "=" * 80)
    print("EXAMPLE: Predicting reward for a state-action pair")
    print("=" * 80)
    example_state = {
        'Bowler Type': 'Fast',
        'Ball Length': 'Good',
        'Ball Line': 'Off',
        'Speed': 140,
        'Field Placement': 'Attacking',
        'Angle': 5,
        'Bounce': 80
    }
    example_action = 'Cover Drive'
    
    print(f"\nState: {example_state}")
    print(f"Action: {example_action}")
    # expected_reward = model.predict_reward(example_state, example_action)
    # print(f"Expected Reward: {expected_reward:.2f}")