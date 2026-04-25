import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
import os

# 1. Helper Function to Load Data
def load_data(filepath="data/telecom_churn.csv"):
    """Loads dataset and prepares features consistent with Petra Telecom tasks."""
    df = pd.read_csv(filepath)
    # Using specific numeric features identified in previous modules
    NUMERIC_FEATURES = ["tenure", "monthly_charges", "total_charges", "num_support_calls", 
                        "senior_citizen", "has_partner", "has_dependents", "contract_months"]
    X = df[NUMERIC_FEATURES]
    y = df['churned']
    # Stratified split to maintain churn class balance
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 2. Helper Function for Nested CV
def run_nested_cv(model_name, estimator, p_grid, X, y):
    """Executes Nested Cross-Validation to address selection bias."""
    # Outer CV: Generalizes model performance evaluation
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123) 
    # Inner CV: Performs hyperparameter tuning
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    inner_scores = []
    outer_scores = []
    
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
        
        # Hyperparameter search within the inner loop
        grid = GridSearchCV(estimator, p_grid, cv=inner_cv, scoring='f1', n_jobs=-1)
        grid.fit(X_train_outer, y_train_outer)
        inner_scores.append(grid.best_score_)
        
        # Final evaluation on unseen data from the outer loop
        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test_outer)
        outer_scores.append(f1_score(y_test_outer, y_pred))
        
    return np.mean(inner_scores), np.mean(outer_scores)

if __name__ == "__main__":
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # --- DATA PREPARATION ---
    X_train, X_test, y_train, y_test = load_data()

    # --- PART 1: GridSearchCV ---
    print("Running Part 1: GridSearchCV (Standard Tuning)...")
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 10, 20, None],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(class_weight='balanced', random_state=42),
        param_grid=rf_params,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Visualization: F1-Score Heatmap
    results_df = pd.DataFrame(grid_search.cv_results_)
    best_min_split = grid_search.best_params_['min_samples_split']
    pivot_table = results_df[results_df['param_min_samples_split'] == best_min_split].pivot(
        index='param_max_depth', columns='param_n_estimators', values='mean_test_score'
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".3f")
    plt.title(f"Random Forest F1-Score (min_samples_split={best_min_split})")
    plt.savefig("results/rf_grid_heatmap.png")
    plt.close()

    print(f"Best Parameters Found: {grid_search.best_params_}")
    print(f"Best Training F1 Score: {grid_search.best_score_:.4f}")

    # --- PART 2: Nested Cross-Validation ---
    print("\nRunning Part 2: Nested Cross-Validation (Unbiased Evaluation)...")

    # Run for Random Forest
    rf_inner, rf_outer = run_nested_cv("RF", RandomForestClassifier(class_weight='balanced', random_state=42), rf_params, X_train, y_train)

    # Run for Decision Tree for comparison
    dt_params = {'max_depth': [3, 5, 10, 20, None], 'min_samples_split': [2, 5, 10]}
    dt_inner, dt_outer = run_nested_cv("DT", DecisionTreeClassifier(class_weight='balanced', random_state=42), dt_params, X_train, y_train)

    # Compilation of Comparison Data
    comparison_data = {
        "Metric": ["Inner best_score (Biased)", "Outer nested score (Honest)", "Selection Bias Gap"],
        "Random Forest": [rf_inner, rf_outer, rf_inner - rf_outer],
        "Decision Tree": [dt_inner, dt_outer, dt_inner - dt_outer]
    }
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv("results/nested_cv_comparison.csv", index=False)

    print("\n=== Comparison Results Summary ===")
    print(comparison_df)
    print("\nTask completed successfully! Files generated in 'results/' directory.")