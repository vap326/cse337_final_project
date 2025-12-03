import pandas as pd
import numpy as np

# --- Configuration ---
DATA_PATH = 'cricket_shot_selection_updated.csv'

def analyze_dataset():
    """
    Loads the cricket dataset and performs the requested statistical analysis:
    1. Total number of rows.
    2. Count of Wicket = 0 (no dismissal).
    3. Percentage analysis of runs scored outcomes.
    """
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}.")
        return

    # 1. Total Number of Rows
    total_rows = len(df)
    
    # 2. Count of Wicket = 0 (No Dismissal)
    wicket_zero_count = df['Wicket'].value_counts().get(0, 0)
    wicket_zero_percent = (wicket_zero_count / total_rows) * 100
    
    # 3. Runs Scored Analysis
    # Create the consolidated 'Outcome' column as defined in train_predictor.py
    def create_outcome(row):
        return 'Wicket' if row['Wicket'] == 1 else str(row['Runs Scored'])

    df['Outcome'] = df.apply(create_outcome, axis=1)

    # Get the value counts for the consolidated outcome
    # Sorting by index ensures 0, 1, 2, 3, 4, 6, Wicket order if they all exist
    outcome_distribution = df['Outcome'].value_counts().sort_index()
    
    # Calculate percentages
    outcome_percent = (outcome_distribution / total_rows) * 100
    
    print("\n--- Dataset Analysis Results ---")
    print(f"Total Rows (Deliveries): {total_rows}")
    print(f"Wicket = 0 Count (No Dismissal): {wicket_zero_count} ({wicket_zero_percent:.2f}%)")
    print("\nConsolidated Outcome Distribution (Runs + Wicket):")
    
    # Combine counts and percentages into a DataFrame for clean output
    summary_df = pd.DataFrame({
        'Count': outcome_distribution,
        'Percentage': outcome_percent.map('{:.2f}%'.format)
    })
    
    # --- FIX: Replace to_markdown() with a loop for universal compatibility ---
    
    # Print header
    print("---------------------------------------")
    print("| Outcome | Count   | Percentage      |")
    print("---------------------------------------")
    
    # Print data rows
    for outcome, row in summary_df.iterrows():
        print(f"| {outcome:<7} | {row['Count']:<7} | {row['Percentage']:<15} |")

    print("---------------------------------------")


if __name__ == '__main__':
    analyze_dataset()