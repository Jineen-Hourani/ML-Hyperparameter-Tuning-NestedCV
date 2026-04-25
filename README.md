
---

# Module 5 Week B — Stretch: Hyperparameter Tuning & Nested CV

## Project Overview
This project focuses on optimizing the **Petra Telecom Churn Prediction** model through systematic hyperparameter tuning and unbiased performance evaluation. 

## Key Implementations
1. **GridSearchCV**: Optimized `RandomForestClassifier` hyperparameters including `n_estimators`, `max_depth`, and `min_samples_split`.
2. **Nested Cross-Validation**: Implemented a nested loop structure (5x5 Stratified CV) to compare `Random Forest` and `Decision Tree` without selection bias.

## Results & Analysis

### 1. Optimal Hyperparameters
After running `GridSearchCV`, the following configuration was identified as the "Champion" model:
* **Best Parameters**: `{'max_depth': 5, 'min_samples_split': 5, 'n_estimators': 50}`.
* **Best F1 Score**: **0.5051**.

### 2. Nested CV & Selection Bias
The comparison between the biased inner loop and the honest outer loop revealed the following:

| Metric | Random Forest | Decision Tree |
| :--- | :--- | :--- |
| **Inner best_score (Biased)** | 0.4959 | 0.4708 |
| **Outer nested score (Honest)** | 0.4986 | 0.4707 |
| **Selection Bias Gap** | -0.0026 | 0.0001 |

**Analysis**:
* **Model Stability**: The "Selection Bias Gap" is extremely low for both models, indicating that our tuning process is robust and the results are not overfitted to the validation folds.
* **Performance**: Random Forest consistently outperformed the Decision Tree in both inner and outer loops, confirming it as the superior choice for this classification task.



## Conclusion
The use of **Nested CV** confirmed that our model's performance is reliable. While the standard Grid Search score was slightly lower than the Nested score for Random Forest, the difference is marginal, providing high confidence for deployment.
