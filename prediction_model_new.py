import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# --- Configuration ---
DATA_PATH = 'cricket_shot_selection_updated.csv'
MODEL_PATH = 'cricket_predictor_model.joblib'
ENCODER_PATH = 'feature_encoder.joblib'

# Define features used for the predictive model (State S + Action A)
STATE_FEATURES = ['Bowler Type', 'Ball Length', 'Ball Line', 'Speed (km/h)', 'Field Placement', 'Angle', 'Bounce (cm)']
ACTION_FEATURE = 'Shot Type'
ALL_FEATURES = STATE_FEATURES + [ACTION_FEATURE]

# Define types of features for the PREDICTIVE MODEL preprocessor
CATEGORICAL_FEATURES = ['Bowler Type', 'Ball Length', 'Ball Line', 'Field Placement', 'Shot Type']
NUMERICAL_FEATURES = ['Speed (km/h)', 'Angle', 'Bounce (cm)']

# Define types of features for the RL OBSERVATION encoder (State only, no action)
STATE_CATEGORICAL_FEATURES = ['Bowler Type', 'Ball Length', 'Ball Line', 'Field Placement']
STATE_NUMERICAL_FEATURES = ['Speed (km/h)', 'Angle', 'Bounce (cm)']

# --- 1. Data Loading and Target Consolidation ---
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}. Please ensure it is in the same directory.")
    exit()

# Target consolidation: If Wicket=1, outcome is 'Wicket', otherwise it's the run count (0-6)
def create_outcome(row):
    return 'Wicket' if row['Wicket'] == 1 else str(row['Runs Scored'])

df['Outcome'] = df.apply(create_outcome, axis=1)

X = df[ALL_FEATURES]
y = df['Outcome']

# --- 2. Create Preprocessing Pipeline for Predictive Model ---
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES),
        ('num', MinMaxScaler(), NUMERICAL_FEATURES)
    ],
    remainder='passthrough'
)

# --- 3. Full Prediction Pipeline ---
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# --- 4. Training ---
print("Starting training of the Predictive Model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

full_pipeline.fit(X_train, y_train)
print("Training complete.")

# --- 5. Evaluation and Verification ---
y_pred = full_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Predictive Model Accuracy (Test Set): {accuracy:.4f}")

# --- 6. Saving Model ---
joblib.dump(full_pipeline, MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")

# --- 7. Create and Save RL Observation Encoder (State Only) ---
# This encoder transforms only the state features (S) into the observation vector (o)
rl_obs_preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), STATE_CATEGORICAL_FEATURES),
        ('num', MinMaxScaler(), STATE_NUMERICAL_FEATURES)
    ],
    remainder='drop'  # Changed from 'passthrough' to 'drop' to be explicit
)

# Fit the preprocessor only on the state features from the training set
rl_obs_preprocessor.fit(X_train[STATE_FEATURES])
joblib.dump(rl_obs_preprocessor, ENCODER_PATH)

print(f"RL Observation Encoder saved to: {ENCODER_PATH}")

# --- 8. Verify Observation Size ---
sample_state = X_train[STATE_FEATURES].head(1)
transformed_sample = rl_obs_preprocessor.transform(sample_state)
observation_size = transformed_sample.shape[1]

print(f"\n{'='*60}")
print(f"CRITICAL: RL Observation Vector Size = {observation_size}")
print(f"{'='*60}")
print("\nBreakdown:")
print(f"  Categorical features (one-hot encoded):")

# Get the feature names after one-hot encoding
try:
    feature_names = rl_obs_preprocessor.get_feature_names_out()
    cat_features = [f for f in feature_names if f.startswith('cat__')]
    num_features = [f for f in feature_names if f.startswith('num__')]
    
    print(f"    - {len(cat_features)} features: {cat_features}")
    print(f"  Numerical features (scaled):")
    print(f"    - {len(num_features)} features: {num_features}")
except:
    # Fallback if get_feature_names_out() doesn't work
    print(f"    - Bowler Type, Ball Length, Ball Line, Field Placement")
    print(f"  Numerical features (scaled):")
    print(f"    - Speed, Angle, Bounce")

print(f"\nUpdate your DQN code with: N_OBSERVATIONS = {observation_size}")