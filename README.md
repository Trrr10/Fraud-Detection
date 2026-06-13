# Behavioural Fraud Detection — Multi-Class XGBoost, Velocity Profiling & SHAP

A research-oriented fraud detection system built on a 555k-row credit card transaction dataset.

## Key Contributions
- Proposed a novel 11-class behavioural fraud taxonomy based on merchant category, transaction amount, time, and geography
- Engineered cardholder velocity features (time-since-last-transaction, spend z-score, implied travel speed) as primary fraud signals
- Trained XGBoost ensemble with Borderline SMOTE to handle severe class imbalance
- Used Isolation Forest anomaly scores as an engineered feature
- Applied SHAP for feature-level explainability and top predictor identification
- Evaluated using AUC-ROC, precision-recall curves, and F1-score

## Tech Stack
Python, XGBoost, Scikit-learn, imbalanced-learn, SHAP, Pandas, NumPy, Matplotlib, Seaborn
