import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix, mean_absolute_error, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns

class CricketRewardPredictionModel:
    """
    Two-stage model to predict cricket rewards:
    Stage 1: Classify outcome type (wicket vs runs) 
    Stage 2: For runs, predict how many (0, 1, 2, 3, 4, 6)
    
    This learns: R(state, action) → reward where reward = -100 for wicket or runs scored
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
        self.y_train_wicket = None
        self.y_test_wicket = None
        self.y_train_runs = None
        self.y_test_runs = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.wicket_model = None  # Stage 1: Wicket classifier
        self.runs_model = None     # Stage 2: Runs classifier
        self.wicket_threshold = 0.5  # Decision threshold for wicket prediction
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
        Preprocess the data and create two targets:
        1. Wicket (binary): Is it a wicket or not?
        2. Runs (multi-class): How many runs (0-6)?
        
        Args:
            runs_column (str): Name of the runs column
            wicket_column (str): Name of the wicket column
        """
        print("\n" + "=" * 80)
        print("PREPROCESSING DATA - TWO-STAGE APPROACH")
        print("=" * 80)
        
        df = self.data.copy()
        df.columns = df.columns.str.strip()
        
        # Find the actual column names
        column_map = {col.lower().replace('_', '').replace(' ', '').replace('(', '').replace(')', ''): col 
                      for col in df.columns}
        
        # Locate runs and wicket columns
        runs_col = None
        wicket_col = None
        
        for possible_name in ['runs', 'run', 'runsscored', 'scored']:
            if possible_name in column_map:
                runs_col = column_map[possible_name]
                break
        
        for possible_name in ['wicket', 'out', 'dismissed', 'iswicket']:
            if possible_name in column_map:
                wicket_col = column_map[possible_name]
                break
        
        if runs_col is None or wicket_col is None:
            print(f"Error: Could not find runs or wicket columns")
            print(f"Available columns: {df.columns.tolist()}")
            return None
        
        print(f"\nUsing '{runs_col}' as runs column")
        print(f"Using '{wicket_col}' as wicket column")
        
        # Map common column name variations for features
        feature_mapping = {
            'bowlertype': ['bowlertype', 'bowler_type', 'bowler type', 'bowler'],
            'length': ['length', 'ball_length', 'ball length', 'balllength', 'ballleng'],
            'line': ['line', 'ball_line', 'ball line', 'ballline', 'balllin'],
            'speed': ['speed', 'ball_speed', 'ball speed', 'velocity', 'speedkmh', 'speedkm/h'],
            'shottype': ['shottype', 'shot_type', 'shot type', 'shot'],
            'fieldplacement': ['fieldplacement', 'field_placement', 'field placement', 'field', 'fieldplacem'],
            'angle': ['angle', 'ball_angle', 'ball angle'],
            'bounce': ['bounce', 'ball_bounce', 'ball bounce', 'bouncecm', 'bouncecm']
        }
        
        actual_columns = {}
        for feature, variations in feature_mapping.items():
            for var in variations:
                if var in column_map:
                    actual_columns[feature] = column_map[var]
                    break
        
        # Manual check for Speed and Bounce with exact column names
        for col in df.columns:
            col_clean = col.lower().replace('_', '').replace(' ', '').replace('(', '').replace(')', '')
            if 'speed' in col_clean and 'speed' not in actual_columns:
                actual_columns['speed'] = col
            if 'bounce' in col_clean and 'bounce' not in actual_columns:
                actual_columns['bounce'] = col
        
        print(f"\nIdentified columns:")
        for feature, col in actual_columns.items():
            print(f"  {feature}: '{col}'")
        
        # Separate state and action features
        action_feature = actual_columns.get('shottype')
        state_features = [col for key, col in actual_columns.items() if key != 'shottype']
        all_features = state_features + ([action_feature] if action_feature else [])
        
        print(f"\nSTATE features (delivery conditions): {state_features}")
        print(f"ACTION feature (shot type): {action_feature}")
        
        # Identify categorical and numerical columns
        categorical_cols = []
        numerical_cols = []
        
        for col in all_features:
            if df[col].dtype == 'object':
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)
        
        print(f"\nCategorical features: {categorical_cols}")
        print(f"Numerical features: {numerical_cols}")
        
        # Encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
            print(f"Encoded '{col}': {len(le.classes_)} unique values")
        
        # Prepare features (X)
        X = df[all_features].copy()
        self.feature_names = all_features
        
        # Normalize numerical features
        if numerical_cols:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
            print(f"\nNormalized {len(numerical_cols)} numerical features")
        
        # Create two separate targets
        y_wicket = df[wicket_col].astype(int)  # Binary: 0 = runs, 1 = wicket
        y_runs = df[runs_col].astype(int)      # Multi-class: 0, 1, 2, 3, 4, 6
        
        print(f"\nTarget 1 - Wicket Distribution:")
        print(y_wicket.value_counts().sort_index())
        print(f"\nTarget 2 - Runs Distribution (for non-wicket deliveries):")
        print(y_runs[y_wicket == 0].value_counts().sort_index())
        
        # Split data - stratify by wicket since it's imbalanced
        self.X_train, self.X_test, self.y_train_wicket, self.y_test_wicket, self.y_train_runs, self.y_test_runs = train_test_split(
            X, y_wicket, y_runs, test_size=0.2, random_state=42, stratify=y_wicket
        )
        
        print(f"\nTraining set size: {self.X_train.shape}")
        print(f"Testing set size: {self.X_test.shape}")
        print(f"\nTraining wicket distribution:")
        print(pd.Series(self.y_train_wicket).value_counts())
        
        return self.X_train, self.X_test, self.y_train_wicket, self.y_test_wicket
    
    def train_models(self, model_type='random_forest', **kwargs):
        """
        Train both models:
        1. Wicket classifier (binary)
        2. Runs classifier (multi-class, for non-wicket deliveries)
        
        Args:
            model_type (str): Type of model ('random_forest' or 'gradient_boosting')
            **kwargs: Additional parameters for the models
        """
        print("\n" + "=" * 80)
        print(f"TRAINING TWO-STAGE MODEL")
        print("=" * 80)
        print("\nStage 1: Wicket Classifier (Binary)")
        print("Stage 2: Runs Classifier (Multi-class)")
        
        # Stage 1: Train wicket classifier with AGGRESSIVE strategies for imbalance
        print("\n" + "-" * 80)
        print("STAGE 1: WICKET CLASSIFIER WITH IMBALANCE HANDLING")
        print("-" * 80)
        
        # Strategy 1: SMOTE-like oversampling of wickets in training
        wicket_mask = self.y_train_wicket == 1
        no_wicket_mask = self.y_train_wicket == 0
        
        n_wickets = wicket_mask.sum()
        n_no_wickets = no_wicket_mask.sum()
        
        print(f"Original training data: {n_no_wickets} no-wickets, {n_wickets} wickets")
        print(f"Imbalance ratio: {n_no_wickets/n_wickets:.1f}:1")
        
        # Oversample wickets by duplicating them
        # Target ratio: 3:1 instead of 18:1
        target_ratio = 3.0
        wicket_multiplier = int(n_no_wickets / (target_ratio * n_wickets))
        
        X_wickets = self.X_train[wicket_mask]
        y_wickets = self.y_train_wicket[wicket_mask]
        
        # Duplicate wicket samples
        X_wickets_oversampled = pd.concat([X_wickets] * wicket_multiplier, ignore_index=True)
        y_wickets_oversampled = pd.concat([y_wickets] * wicket_multiplier, ignore_index=True)
        
        # Combine with no-wicket samples
        X_train_balanced = pd.concat([
            self.X_train[no_wicket_mask],
            X_wickets_oversampled
        ], ignore_index=True)
        
        y_train_balanced = pd.concat([
            self.y_train_wicket[no_wicket_mask],
            y_wickets_oversampled
        ], ignore_index=True)
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(X_train_balanced))
        X_train_balanced = X_train_balanced.iloc[shuffle_idx].reset_index(drop=True)
        y_train_balanced = y_train_balanced.iloc[shuffle_idx].reset_index(drop=True)
        
        print(f"\nAfter oversampling: {(y_train_balanced==0).sum()} no-wickets, {(y_train_balanced==1).sum()} wickets")
        print(f"New ratio: {(y_train_balanced==0).sum()/(y_train_balanced==1).sum():.1f}:1")
        
        # Strategy 2: Further boost wicket class with weights
        wicket_counts = pd.Series(y_train_balanced).value_counts()
        class_weights = {0: 1.0, 1: 2.0}  # Give wickets 2x more weight even after oversampling
        print(f"Additional class weights: No Wicket: 1.0, Wicket: 2.0")
        
        if model_type == 'random_forest':
            self.wicket_model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 400),
                max_depth=kwargs.get('max_depth', 15),
                min_samples_split=kwargs.get('min_samples_split', 10),
                min_samples_leaf=kwargs.get('min_samples_leaf', 5),
                max_features='sqrt',
                class_weight=class_weights,
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
        elif model_type == 'gradient_boosting':
            self.wicket_model = GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 400),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.05),
                subsample=0.8,
                random_state=42,
                verbose=1
            )
        
        print("\nTraining wicket classifier on balanced data...")
        self.wicket_model.fit(X_train_balanced, y_train_balanced)
        print("✓ Wicket classifier training complete!")
        
        # Evaluate on original unbalanced test set
        y_test_wicket_pred = self.wicket_model.predict(self.X_test)
        wicket_recall = np.sum((self.y_test_wicket == 1) & (y_test_wicket_pred == 1)) / np.sum(self.y_test_wicket == 1)
        print(f"\nWicket Recall on Test Set: {wicket_recall:.2%}")
        
        # If still too low, adjust decision threshold
        if wicket_recall < 0.3:
            print("\nWicket recall too low! Adjusting decision threshold...")
            y_test_proba = self.wicket_model.predict_proba(self.X_test)[:, 1]
            
            # Find threshold that gives ~50% recall
            thresholds = np.linspace(0.1, 0.9, 50)
            best_threshold = 0.5
            best_f1 = 0
            
            for thresh in thresholds:
                y_pred_thresh = (y_test_proba >= thresh).astype(int)
                recall = np.sum((self.y_test_wicket == 1) & (y_pred_thresh == 1)) / np.sum(self.y_test_wicket == 1)
                precision = np.sum((self.y_test_wicket == 1) & (y_pred_thresh == 1)) / max(np.sum(y_pred_thresh == 1), 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-10)
                
                if recall > 0.4 and f1 > best_f1:  # Prioritize recall
                    best_f1 = f1
                    best_threshold = thresh
            
            self.wicket_threshold = best_threshold
            print(f"Optimal threshold: {best_threshold:.3f}")
            
            y_test_wicket_pred_adjusted = (y_test_proba >= best_threshold).astype(int)
            wicket_recall_adjusted = np.sum((self.y_test_wicket == 1) & (y_test_wicket_pred_adjusted == 1)) / np.sum(self.y_test_wicket == 1)
            print(f"Adjusted Wicket Recall: {wicket_recall_adjusted:.2%}")
        else:
            self.wicket_threshold = 0.5
        
        # Stage 2: Train runs classifier (only on non-wicket deliveries)
        print("\n" + "-" * 80)
        print("STAGE 2: RUNS CLASSIFIER")
        print("-" * 80)
        
        # Filter to only non-wicket deliveries
        non_wicket_mask_train = self.y_train_wicket == 0
        X_train_runs = self.X_train[non_wicket_mask_train]
        y_train_runs = self.y_train_runs[non_wicket_mask_train]
        
        print(f"Training on {len(X_train_runs)} non-wicket deliveries")
        print(f"Runs distribution: {pd.Series(y_train_runs).value_counts().sort_index().to_dict()}")
        
        if model_type == 'random_forest':
            self.runs_model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 400),
                max_depth=kwargs.get('max_depth', 25),
                min_samples_split=kwargs.get('min_samples_split', 5),
                min_samples_leaf=kwargs.get('min_samples_leaf', 2),
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=1
            )
        elif model_type == 'gradient_boosting':
            self.runs_model = GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 400),
                max_depth=kwargs.get('max_depth', 8),
                learning_rate=kwargs.get('learning_rate', 0.05),
                subsample=0.8,
                random_state=42,
                verbose=1
            )
        
        print("\nTraining runs classifier...")
        self.runs_model.fit(X_train_runs, y_train_runs)
        print("✓ Runs classifier training complete!")
        
        return self.wicket_model, self.runs_model
    
    def predict_reward(self, X):
        """
        Predict reward using two-stage approach with adjusted threshold.
        
        Args:
            X: Features (can be single sample or batch)
            
        Returns:
            array: Predicted rewards (-100 for wicket, or runs 0-6)
        """
        # Stage 1: Predict wicket or not using adjusted threshold
        wicket_proba = self.wicket_model.predict_proba(X)[:, 1]
        wicket_pred = (wicket_proba >= self.wicket_threshold).astype(int)
        
        # Stage 2: For non-wickets, predict runs
        rewards = np.full(len(X), self.wicket_penalty, dtype=float)
        non_wicket_mask = wicket_pred == 0
        
        if non_wicket_mask.sum() > 0:
            runs_pred = self.runs_model.predict(X[non_wicket_mask])
            rewards[non_wicket_mask] = runs_pred
        
        return rewards
    
    def evaluate_models(self, plot=True):
        """
        Comprehensive evaluation of both models.
        
        Args:
            plot (bool): Whether to generate visualization plots
        """
        print("\n" + "=" * 80)
        print("MODEL EVALUATION - TWO-STAGE APPROACH")
        print("=" * 80)
        
        # Stage 1 Evaluation: Wicket Prediction
        print("\n" + "-" * 80)
        print("STAGE 1: WICKET CLASSIFIER PERFORMANCE")
        print("-" * 80)
        
        y_train_wicket_pred = self.wicket_model.predict(self.X_train)
        y_test_wicket_pred = self.wicket_model.predict(self.X_test)
        
        train_wicket_acc = accuracy_score(self.y_train_wicket, y_train_wicket_pred)
        test_wicket_acc = accuracy_score(self.y_test_wicket, y_test_wicket_pred)
        
        train_wicket_f1 = f1_score(self.y_train_wicket, y_train_wicket_pred, average='binary')
        test_wicket_f1 = f1_score(self.y_test_wicket, y_test_wicket_pred, average='binary')
        
        print(f"\n{'Metric':<30} {'Training':<15} {'Testing':<15}")
        print("-" * 60)
        print(f"{'Wicket Accuracy':<30} {train_wicket_acc:<15.4f} {test_wicket_acc:<15.4f}")
        print(f"{'Wicket F1-Score':<30} {train_wicket_f1:<15.4f} {test_wicket_f1:<15.4f}")
        
        print("\nWicket Classification Report (Test Set):")
        print(classification_report(self.y_test_wicket, y_test_wicket_pred, 
                                   target_names=['No Wicket', 'Wicket']))
        
        # Stage 2 Evaluation: Runs Prediction (on non-wickets only)
        print("\n" + "-" * 80)
        print("STAGE 2: RUNS CLASSIFIER PERFORMANCE")
        print("-" * 80)
        
        non_wicket_mask_train = self.y_train_wicket == 0
        non_wicket_mask_test = self.y_test_wicket == 0
        
        X_train_runs = self.X_train[non_wicket_mask_train]
        y_train_runs_actual = self.y_train_runs[non_wicket_mask_train]
        y_train_runs_pred = self.runs_model.predict(X_train_runs)
        
        X_test_runs = self.X_test[non_wicket_mask_test]
        y_test_runs_actual = self.y_test_runs[non_wicket_mask_test]
        y_test_runs_pred = self.runs_model.predict(X_test_runs)
        
        train_runs_acc = accuracy_score(y_train_runs_actual, y_train_runs_pred)
        test_runs_acc = accuracy_score(y_test_runs_actual, y_test_runs_pred)
        
        train_runs_mae = mean_absolute_error(y_train_runs_actual, y_train_runs_pred)
        test_runs_mae = mean_absolute_error(y_test_runs_actual, y_test_runs_pred)
        
        print(f"\n{'Metric':<30} {'Training':<15} {'Testing':<15}")
        print("-" * 60)
        print(f"{'Runs Accuracy':<30} {train_runs_acc:<15.4f} {test_runs_acc:<15.4f}")
        print(f"{'Runs MAE':<30} {train_runs_mae:<15.4f} {test_runs_mae:<15.4f}")
        
        print("\nRuns Classification Report (Test Set, Non-Wickets Only):")
        print(classification_report(y_test_runs_actual, y_test_runs_pred))
        
        # Overall Reward Prediction
        print("\n" + "-" * 80)
        print("OVERALL REWARD PREDICTION PERFORMANCE")
        print("-" * 80)
        
        y_train_reward_pred = self.predict_reward(self.X_train)
        y_test_reward_pred = self.predict_reward(self.X_test)
        
        # Create actual rewards
        y_train_reward_actual = np.where(self.y_train_wicket == 1, 
                                         self.wicket_penalty, 
                                         self.y_train_runs)
        y_test_reward_actual = np.where(self.y_test_wicket == 1, 
                                        self.wicket_penalty, 
                                        self.y_test_runs)
        
        train_reward_acc = accuracy_score(y_train_reward_actual, y_train_reward_pred)
        test_reward_acc = accuracy_score(y_test_reward_actual, y_test_reward_pred)
        
        train_reward_mae = mean_absolute_error(y_train_reward_actual, y_train_reward_pred)
        test_reward_mae = mean_absolute_error(y_test_reward_actual, y_test_reward_pred)
        
        print(f"\n{'Metric':<30} {'Training':<15} {'Testing':<15}")
        print("-" * 60)
        print(f"{'Overall Accuracy':<30} {train_reward_acc:<15.4f} {test_reward_acc:<15.4f}")
        print(f"{'Overall MAE':<30} {train_reward_mae:<15.4f} {test_reward_mae:<15.4f}")
        
        # Per-reward accuracy
        print("\nPer-Reward Type Accuracy (Test Set):")
        unique_rewards = sorted(np.unique(y_test_reward_actual))
        for reward in unique_rewards:
            mask = y_test_reward_actual == reward
            if mask.sum() > 0:
                acc = accuracy_score(y_test_reward_actual[mask], y_test_reward_pred[mask])
                count = mask.sum()
                print(f"  Reward {reward:>4}: {acc:.4f} ({acc*100:5.1f}%) - {count} samples")
        
        # Feature importance
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE")
        print("=" * 80)
        
        if hasattr(self.wicket_model, 'feature_importances_'):
            wicket_importance = pd.DataFrame({
                'Feature': self.feature_names,
                'Wicket_Model': self.wicket_model.feature_importances_,
                'Runs_Model': self.runs_model.feature_importances_
            }).sort_values('Wicket_Model', ascending=False)
            print("\nTop features for each model:")
            print(wicket_importance.to_string(index=False))
        
        if plot:
            self._plot_evaluation(y_test_wicket_pred, y_test_runs_actual, y_test_runs_pred,
                                 y_test_reward_actual, y_test_reward_pred, wicket_importance)
        
        return {
            'test_wicket_acc': test_wicket_acc,
            'test_wicket_f1': test_wicket_f1,
            'test_runs_acc': test_runs_acc,
            'test_runs_mae': test_runs_mae,
            'test_overall_acc': test_reward_acc,
            'test_overall_mae': test_reward_mae
        }
    
    def _plot_evaluation(self, y_test_wicket_pred, y_test_runs_actual, y_test_runs_pred,
                        y_test_reward_actual, y_test_reward_pred, feature_importance):
        """Generate comprehensive evaluation plots."""
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Wicket Confusion Matrix
        plt.subplot(3, 4, 1)
        cm_wicket = confusion_matrix(self.y_test_wicket, y_test_wicket_pred)
        sns.heatmap(cm_wicket, annot=True, fmt='d', cmap='Blues')
        plt.title('Wicket Prediction\nConfusion Matrix', fontsize=11, fontweight='bold')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.xticks([0.5, 1.5], ['No Wicket', 'Wicket'])
        plt.yticks([0.5, 1.5], ['No Wicket', 'Wicket'])
        
        # 2. Runs Confusion Matrix
        plt.subplot(3, 4, 2)
        cm_runs = confusion_matrix(y_test_runs_actual, y_test_runs_pred)
        sns.heatmap(cm_runs, annot=True, fmt='d', cmap='Greens', cbar_kws={'label': 'Count'})
        plt.title('Runs Prediction\nConfusion Matrix', fontsize=11, fontweight='bold')
        plt.ylabel('Actual Runs')
        plt.xlabel('Predicted Runs')
        
        # 3. Overall Reward Distribution
        plt.subplot(3, 4, 3)
        all_rewards = sorted(set(y_test_reward_actual) | set(y_test_reward_pred))
        actual_counts = [np.sum(y_test_reward_actual == r) for r in all_rewards]
        pred_counts = [np.sum(y_test_reward_pred == r) for r in all_rewards]
        
        x = np.arange(len(all_rewards))
        width = 0.35
        plt.bar(x - width/2, actual_counts, width, label='Actual', alpha=0.8)
        plt.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
        plt.xlabel('Reward')
        plt.ylabel('Count')
        plt.title('Reward Distribution\n(Actual vs Predicted)', fontsize=11, fontweight='bold')
        plt.xticks(x, all_rewards, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # 4. Per-Reward Accuracy
        plt.subplot(3, 4, 4)
        unique_rewards = sorted(np.unique(y_test_reward_actual))
        accuracies = []
        for reward in unique_rewards:
            mask = y_test_reward_actual == reward
            if mask.sum() > 0:
                acc = accuracy_score(y_test_reward_actual[mask], y_test_reward_pred[mask])
                accuracies.append(acc)
        
        colors = ['red' if r == self.wicket_penalty else 'green' for r in unique_rewards]
        plt.bar(range(len(unique_rewards)), accuracies, color=colors, alpha=0.7)
        plt.xticks(range(len(unique_rewards)), unique_rewards)
        plt.ylabel('Accuracy')
        plt.xlabel('Reward Value')
        plt.title('Per-Reward Accuracy', fontsize=11, fontweight='bold')
        plt.ylim([0, 1])
        plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 5. Feature Importance - Wicket Model
        plt.subplot(3, 4, 5)
        if feature_importance is not None:
            top_wicket = feature_importance.nlargest(8, 'Wicket_Model')
            plt.barh(range(len(top_wicket)), top_wicket['Wicket_Model'], color='steelblue')
            plt.yticks(range(len(top_wicket)), top_wicket['Feature'])
            plt.xlabel('Importance')
            plt.title('Feature Importance\n(Wicket Model)', fontsize=11, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
        
        # 6. Feature Importance - Runs Model
        plt.subplot(3, 4, 6)
        if feature_importance is not None:
            top_runs = feature_importance.nlargest(8, 'Runs_Model')
            plt.barh(range(len(top_runs)), top_runs['Runs_Model'], color='forestgreen')
            plt.yticks(range(len(top_runs)), top_runs['Feature'])
            plt.xlabel('Importance')
            plt.title('Feature Importance\n(Runs Model)', fontsize=11, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
        
        # 7. Prediction Error Distribution
        plt.subplot(3, 4, 7)
        errors = y_test_reward_actual - y_test_reward_pred
        plt.hist(errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Error Distribution', fontsize=11, fontweight='bold')
        plt.axvline(x=0, color='r', linestyle='--', linewidth=2)
        plt.grid(True, alpha=0.3)
        
        # 8. Actual vs Predicted Scatter
        plt.subplot(3, 4, 8)
        # Add jitter for visibility
        jitter_amount = 0.3
        x_jitter = y_test_reward_actual + np.random.uniform(-jitter_amount, jitter_amount, len(y_test_reward_actual))
        y_jitter = y_test_reward_pred + np.random.uniform(-jitter_amount, jitter_amount, len(y_test_reward_pred))
        plt.scatter(x_jitter, y_jitter, alpha=0.3, s=20)
        
        min_val = min(y_test_reward_actual.min(), y_test_reward_pred.min())
        max_val = max(y_test_reward_actual.max(), y_test_reward_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
        plt.xlabel('Actual Reward')
        plt.ylabel('Predicted Reward')
        plt.title('Actual vs Predicted', fontsize=11, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 9. Wicket Prediction by True Label
        plt.subplot(3, 4, 9)
        wicket_data = pd.DataFrame({
            'Actual': self.y_test_wicket,
            'Predicted': y_test_wicket_pred
        })
        wicket_crosstab = pd.crosstab(wicket_data['Actual'], wicket_data['Predicted'], normalize='index')
        wicket_crosstab.plot(kind='bar', stacked=False, ax=plt.gca(), color=['lightcoral', 'lightblue'])
        plt.xlabel('Actual Wicket')
        plt.ylabel('Proportion')
        plt.title('Wicket Prediction\nby True Label', fontsize=11, fontweight='bold')
        plt.xticks([0, 1], ['No Wicket', 'Wicket'], rotation=0)
        plt.legend(['Pred: No Wicket', 'Pred: Wicket'])
        plt.grid(True, alpha=0.3, axis='y')
        
        # 10. Runs Distribution Comparison
        plt.subplot(3, 4, 10)
        runs_comparison = pd.DataFrame({
            'Actual': y_test_runs_actual,
            'Predicted': y_test_runs_pred
        })
        actual_dist = runs_comparison['Actual'].value_counts().sort_index()
        pred_dist = runs_comparison['Predicted'].value_counts().sort_index()
        
        x_pos = np.arange(len(actual_dist))
        width = 0.35
        plt.bar(x_pos - width/2, actual_dist.values, width, label='Actual', alpha=0.8)
        plt.bar(x_pos + width/2, pred_dist.values, width, label='Predicted', alpha=0.8)
        plt.xlabel('Runs')
        plt.ylabel('Count')
        plt.title('Runs Distribution\n(Non-Wickets Only)', fontsize=11, fontweight='bold')
        plt.xticks(x_pos, actual_dist.index)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # 11. Cumulative Error
        plt.subplot(3, 4, 11)
        sorted_abs_errors = np.sort(np.abs(errors))
        cumulative = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors)
        plt.plot(sorted_abs_errors, cumulative, linewidth=2, color='purple')
        plt.xlabel('Absolute Error')
        plt.ylabel('Cumulative Proportion')
        plt.title('Cumulative Error\nDistribution', fontsize=11, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
        plt.axhline(y=0.9, color='gray', linestyle='--', linewidth=1)
        
        # 12. Model Performance Summary
        plt.subplot(3, 4, 12)
        plt.axis('off')
        
        wicket_acc = accuracy_score(self.y_test_wicket, y_test_wicket_pred)
        runs_acc = accuracy_score(y_test_runs_actual, y_test_runs_pred)
        overall_acc = accuracy_score(y_test_reward_actual, y_test_reward_pred)
        overall_mae = mean_absolute_error(y_test_reward_actual, y_test_reward_pred)
        
        # Calculate wicket-specific metrics
        wicket_mask = self.y_test_wicket == 1
        if wicket_mask.sum() > 0:
            wicket_specific_acc = accuracy_score(
                y_test_reward_actual[wicket_mask], 
                y_test_reward_pred[wicket_mask]
            )
        else:
            wicket_specific_acc = 0.0
        
        summary_text = f"""
MODEL PERFORMANCE SUMMARY
{'='*30}

Stage 1: Wicket Classification
  Accuracy: {wicket_acc:.2%}
  
Stage 2: Runs Classification
  Accuracy: {runs_acc:.2%}
  
Overall Reward Prediction
  Accuracy: {overall_acc:.2%}
  MAE: {overall_mae:.2f}
  
Wicket Prediction Accuracy: {wicket_specific_acc:.2%}

Total Test Samples: {len(y_test_reward_actual)}
  Wickets: {wicket_mask.sum()}
  Runs: {(~wicket_mask).sum()}
        """
        
        plt.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig('two_stage_model_evaluation.png', dpi=300, bbox_inches='tight')
        print("\n✓ Evaluation plots saved as 'two_stage_model_evaluation.png'")
        plt.show()
    
    def predict_reward_single(self, state, action):
        """
        Predict the expected reward for a given state-action pair.
        
        Args:
            state (dict): Delivery conditions (Bowler Type, Line, Length, Speed, etc.)
            action (str): Shot type
            
        Returns:
            float: Expected reward (runs or wicket penalty)
        """
        # Combine state and action
        features = {**state}
        
        # Find the shot type column name
        shot_col = None
        for col in self.feature_names:
            if 'shot' in col.lower():
                shot_col = col
                break
        
        if shot_col:
            features[shot_col] = action
        
        df = pd.DataFrame([features])
        
        # Encode categorical features
        for col in self.label_encoders.keys():
            if col in df.columns:
                try:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                except ValueError:
                    # Unknown category, use most frequent
                    df[col] = 0
        
        # Normalize numerical features
        numerical_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
        if numerical_cols:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        # Predict using two-stage approach
        reward = self.predict_reward(df[self.feature_names].values)
        return reward[0]
    
    def get_reward_probabilities(self, state, action):
        """
        Get probability distribution over all possible rewards.
        Useful for stochastic RL environments.
        
        Args:
            state (dict): Delivery conditions
            action (str): Shot type
            
        Returns:
            dict: {reward_value: probability}
        """
        # Prepare features
        features = {**state}
        shot_col = None
        for col in self.feature_names:
            if 'shot' in col.lower():
                shot_col = col
                break
        if shot_col:
            features[shot_col] = action
        
        df = pd.DataFrame([features])
        
        # Encode and normalize
        for col in self.label_encoders.keys():
            if col in df.columns:
                try:
                    df[col] = self.label_encoders[col].transform(df[col].astype(str))
                except ValueError:
                    df[col] = 0
        
        numerical_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
        if numerical_cols:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        X = df[self.feature_names].values
        
        # Get probabilities from both models
        wicket_proba = self.wicket_model.predict_proba(X)[0]  # [P(no_wicket), P(wicket)]
        runs_proba = self.runs_model.predict_proba(X)[0]      # [P(0), P(1), P(2), ...]
        
        # Combine probabilities
        reward_probs = {}
        
        # Probability of wicket
        reward_probs[self.wicket_penalty] = wicket_proba[1]
        
        # Probabilities of different run outcomes
        runs_classes = self.runs_model.classes_
        for i, runs in enumerate(runs_classes):
            reward_probs[runs] = wicket_proba[0] * runs_proba[i]
        
        return reward_probs


# Example usage for RL integration
if __name__ == "__main__":
    # Initialize model
    model = CricketRewardPredictionModel('cricket_shot_selection_updated.csv', wicket_penalty=-100)
    
    # Step 1: Load and explore data
    print("\nSTEP 1: DATA LOADING")
    data = model.load_and_explore_data()
    
    # Step 2: Preprocess data
    print("\nSTEP 2: DATA PREPROCESSING")
    X_train, X_test, y_train, y_test = model.preprocess_data(
        runs_column='Runs Scored',
        wicket_column='Wicket'
    )
    
    # Step 3: Train both models
    print("\nSTEP 3: MODEL TRAINING")
    wicket_model, runs_model = model.train_models(
        model_type='random_forest',
        n_estimators=300,
        max_depth=20
    )
    
    # Step 4: Evaluate models
    print("\nSTEP 4: MODEL EVALUATION")
    metrics = model.evaluate_models(plot=True)
    
    print("\n" + "=" * 80)
    print("TWO-STAGE REWARD PREDICTION MODEL COMPLETE!")
    print("=" * 80)
    print(f"\nWicket Classification Accuracy: {metrics['test_wicket_acc']:.2%}")
    print(f"Runs Classification Accuracy: {metrics['test_runs_acc']:.2%}")
    print(f"Overall Reward Accuracy: {metrics['test_overall_acc']:.2%}")
    print(f"Overall MAE: {metrics['test_overall_mae']:.2f}")
    
    print(f"\nModel ready for RL environment!")
    print(f"   Use model.predict_reward_single(state, action) to get single predictions")
    print(f"   Use model.get_reward_probabilities(state, action) for stochastic sampling")
    
    # Example prediction
    print("\n" + "=" * 80)
    print("EXAMPLE: Predicting reward for a state-action pair")
    print("=" * 80)
    
    example_state = {
        'Bowler Type': 'Fast',
        'Ball Length': 'Good',
        'Ball Line': 'Off-stump',
        'Speed (km/h)': 140,
        'Field Placement': 'Defensive',
        'Angle': 5.0,
        'Bounce (cm)': 70
    }
    example_action = 'Cover Drive'
    
    print(f"\nState (Delivery Conditions):")
    for key, val in example_state.items():
        print(f"  {key}: {val}")
    print(f"\nAction (Shot Type): {example_action}")
    
    # Get deterministic prediction
    expected_reward = model.predict_reward_single(example_state, example_action)
    print(f"\nExpected Reward: {expected_reward:.1f}")
    
    # Get probability distribution
    reward_probs = model.get_reward_probabilities(example_state, example_action)
    print(f"\nReward Probability Distribution:")
    for reward, prob in sorted(reward_probs.items()):
        if prob > 0.01:  # Only show rewards with >1% probability
            print(f"  Reward {reward:>4}: {prob:.1%}")
    
    print("\n" + "=" * 80)
    print("Integration with RL Environment:")
    print("=" * 80)
    print("""
# In your RL environment's step function:

def step(self, action):
    state = self.get_current_state()
    
    # Deterministic reward (for DQN)
    reward = model.predict_reward_single(state, action)
    
    # OR stochastic reward (more realistic)
    reward_probs = model.get_reward_probabilities(state, action)
    reward = np.random.choice(
        list(reward_probs.keys()), 
        p=list(reward_probs.values())
    )
    
    done = (reward == -100)  # Episode ends on wicket
    next_state = self.generate_next_delivery()
    
    return next_state, reward, done, {}
    """)