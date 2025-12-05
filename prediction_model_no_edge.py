import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, f1_score, balanced_accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns

# --- Configuration ---
DATA_PATH = 'cricket_shot_selection_updated.csv'
MODEL_PATH = 'cricket_predictor_model.joblib'
ENCODER_PATH = 'feature_encoder.joblib'

STATE_FEATURES = ['Bowler Type', 'Ball Length', 'Ball Line', 'Speed (km/h)', 
                  'Field Placement', 'Angle', 'Bounce (cm)']
ACTION_FEATURE = 'Shot Type'
ALL_FEATURES = STATE_FEATURES + [ACTION_FEATURE]

CATEGORICAL_FEATURES = ['Bowler Type', 'Ball Length', 'Ball Line', 'Field Placement', 'Shot Type']
NUMERICAL_FEATURES = ['Speed (km/h)', 'Angle', 'Bounce (cm)']

STATE_CATEGORICAL_FEATURES = ['Bowler Type', 'Ball Length', 'Ball Line', 'Field Placement']
STATE_NUMERICAL_FEATURES = ['Speed (km/h)', 'Angle', 'Bounce (cm)']

print("="*80)
print("CRICKET PREDICTIVE MODEL TRAINING - EDGE REMOVED")
print("="*80)

# Load data
df = pd.read_csv(DATA_PATH)
print(f"\n✓ Loaded {len(df)} samples")

# ==============================================================================
# REMOVE EDGE FROM DATA
# ==============================================================================

print("\n" + "="*80)
print("REMOVING 'EDGE' SHOTS FROM DATASET")
print("="*80)

original_count = len(df)
edge_count = (df['Shot Type'] == 'Edge').sum()

print(f"\nOriginal dataset: {original_count} samples")
print(f"Edge shots: {edge_count} ({edge_count/original_count*100:.1f}%)")

# Remove Edge shots
df = df[df['Shot Type'] != 'Edge'].copy()

print(f"After removal: {len(df)} samples")
print(f"Removed: {edge_count} samples")
print("\n✓ Edge shots removed from training data")

print("\nRemaining shot types:")
remaining_shots = df['Shot Type'].value_counts()
for shot, count in remaining_shots.items():
    print(f"  {shot:<20} {count:4d} ({count/len(df)*100:5.1f}%)")

# Create outcome
def create_outcome(row):
    return 'Wicket' if row['Wicket'] == 1 else str(int(row['Runs Scored']))

df['Outcome'] = df.apply(create_outcome, axis=1)

# ==============================================================================
# DATA DISTRIBUTION ANALYSIS
# ==============================================================================

print("\n" + "="*80)
print("DATA DISTRIBUTION ANALYSIS (AFTER EDGE REMOVAL)")
print("="*80)

outcome_counts = df['Outcome'].value_counts().sort_index()
print("\nOutcome Distribution:")
for outcome in ['0', '1', '2', '3', '4', '6', 'Wicket']:
    count = outcome_counts.get(outcome, 0)
    percentage = (count / len(df)) * 100
    print(f"  {outcome:7s}: {count:4d} ({percentage:5.1f}%)")

total_wickets = outcome_counts.get('Wicket', 0)
wicket_rate = (total_wickets / len(df)) * 100
print(f"\nWicket rate: {wicket_rate:.1f}%")

# Shot type vs outcome
print("\n" + "="*80)
print("SHOT TYPE ANALYSIS (WITHOUT EDGE)")
print("="*80)

print("\nWicket rates by shot type:")
for shot in df['Shot Type'].unique():
    shot_data = df[df['Shot Type'] == shot]
    wicket_count = (shot_data['Outcome'] == 'Wicket').sum()
    wicket_pct = (wicket_count / len(shot_data)) * 100
    
    # Calculate EV
    ev = 0
    for _, row in shot_data.iterrows():
        if row['Outcome'] == 'Wicket':
            ev += -100
        else:
            ev += int(row['Outcome'])
    ev /= len(shot_data)
    
    print(f"  {shot:<20} Wicket: {wicket_pct:5.1f}%, EV: {ev:6.2f}")

# ==============================================================================
# TRAIN-TEST SPLIT
# ==============================================================================

X = df[ALL_FEATURES]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Train set: {len(X_train)} samples")
print(f"✓ Test set:  {len(X_test)} samples")

# ==============================================================================
# PREPROCESSING AND TRAINING
# ==============================================================================

print("\n" + "="*80)
print("MODEL TRAINING")
print("="*80)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
         CATEGORICAL_FEATURES),
        ('num', StandardScaler(), NUMERICAL_FEATURES)
    ],
    remainder='drop'
)

# Random Forest
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', rf_model)
])

rf_pipeline.fit(X_train, y_train)
print("✓ Random Forest trained")

# Gradient Boosting
print("\nTraining Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=7,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=42
)

gb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', gb_model)
])

gb_pipeline.fit(X_train, y_train)
print("✓ Gradient Boosting trained")

# ==============================================================================
# EVALUATION
# ==============================================================================

print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

models = {
    'Random Forest': rf_pipeline,
    'Gradient Boosting': gb_pipeline
}

results = {}

for name, model in models.items():
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    results[name] = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'predictions': y_pred
    }
    
    print(f"\n{name}:")
    print(f"  Accuracy:         {accuracy:.4f}")
    print(f"  F1-Score (Macro): {f1_macro:.4f}")
    
    # Per-class accuracy
    print(f"\n  Per-Class Accuracy:")
    for outcome in ['0', '1', '2', '3', '4', '6', 'Wicket']:
        mask = y_test == outcome
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == y_test[mask]).sum() / mask.sum()
            print(f"    {outcome:7s}: {class_acc:.4f} ({mask.sum()} samples)")

best_name = max(results.keys(), key=lambda k: results[k]['f1_macro'])
best_model = models[best_name]
best_score = results[best_name]['f1_macro']

print(f"\n{'='*80}")
print(f"BEST MODEL: {best_name}")
print(f"  F1-Score (Macro): {best_score:.4f}")
print(f"  Accuracy: {results[best_name]['accuracy']:.4f}")
print(f"{'='*80}")

# Classification report
y_pred_best = results[best_name]['predictions']
print("\n" + "="*80)
print("CLASSIFICATION REPORT")
print("="*80)
print(classification_report(y_test, y_pred_best))

# Confusion matrix
outcomes = ['0', '1', '2', '3', '4', '6', 'Wicket']
cm = confusion_matrix(y_test, y_pred_best, labels=outcomes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=outcomes, yticklabels=outcomes)
plt.title(f'{best_name} - Confusion Matrix (Edge Removed)', fontweight='bold')
plt.ylabel('True')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('predictor_confusion_matrix_no_edge.png', dpi=300)
print("\n✓ Confusion matrix saved to: predictor_confusion_matrix_no_edge.png")
plt.close()

# ==============================================================================
# SAVE MODEL
# ==============================================================================

joblib.dump(best_model, MODEL_PATH)
print(f"\n✓ Best model saved to: {MODEL_PATH}")

# ==============================================================================
# RL ENCODER
# ==============================================================================

print("\n" + "="*80)
print("RL OBSERVATION ENCODER")
print("="*80)

rl_obs_preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), 
         STATE_CATEGORICAL_FEATURES),
        ('num', StandardScaler(), STATE_NUMERICAL_FEATURES)
    ],
    remainder='drop'
)

rl_obs_preprocessor.fit(X_train[STATE_FEATURES])
joblib.dump(rl_obs_preprocessor, ENCODER_PATH)

observation_size = rl_obs_preprocessor.transform(X_train[STATE_FEATURES].head(1)).shape[1]

print(f"\n✓ RL Encoder saved: N_OBSERVATIONS = {observation_size}")

# ==============================================================================
# UPDATE ACTION MAP FOR DQN
# ==============================================================================

print("\n" + "="*80)
print("⚠️  CRITICAL: UPDATE YOUR DQN CODE")
print("="*80)

remaining_shot_types = sorted(df['Shot Type'].unique())
print("\nNew ACTION_MAP (Edge removed):")
print("ACTION_MAP = {")
for i, shot in enumerate(remaining_shot_types):
    print(f"    {i}: '{shot}',")
print("}")
print(f"\nN_ACTIONS = {len(remaining_shot_types)}")
print(f"N_OBSERVATIONS = {observation_size}")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\n✓ Edge shots removed from action space")
print("✓ Model trained on 8 legitimate shot types")
print("✓ Update DQN code with new ACTION_MAP and N_ACTIONS")
print("="*80)