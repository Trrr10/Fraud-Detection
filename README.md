# Explainable Multi-Class Fraud Detection — Behavioral & Velocity Feature Engineering

A research-oriented fraud detection system built on a real-world credit card transaction dataset, focused on **severe class imbalance, behavioral feature engineering, and model explainability** rather than headline accuracy.

## Overview
This project classifies credit card transactions into an **11-class behavioral fraud taxonomy** (10 fraud sub-types + legitimate), using engineered cardholder-level, velocity, and geographic features. The dataset has a **231:1 class imbalance** (240,506 transactions, only 1,035 fraudulent), which shaped every modeling and evaluation decision below.

## Key Contributions
- Proposed an 11-class behavioral fraud taxonomy (e.g., merchant category, transaction amount, time-of-day, geographic anomalies) instead of simple binary fraud/legitimate labeling
- Engineered **causal, point-in-time cardholder features** — expanding mean/std of spend, transaction velocity, category/merchant diversity, and geographic "implied speed" — computed strictly from each cardholder's transaction history *up to that point in time* (data sorted by cardholder + timestamp before feature construction, avoiding lookahead leakage)
- Validated feature significance using **Mann–Whitney U testing** for fraud vs. legitimate transaction amounts, and a chi-squared test for merchant category association
- Handled severe imbalance using **Borderline-SMOTE**, applied only to the training split (post train/test split) to avoid contaminating the test set
- Trained and compared **XGBoost, LightGBM, and Random Forest** ensembles, with **Isolation Forest anomaly scores** as an additional engineered feature
- Applied **SHAP** for global and per-class feature-level explainability
- Ran an **ablation study** on the engineered velocity features to test whether they actually improved model performance

## Evaluation Approach
With a 231:1 imbalance, accuracy is not a meaningful metric — a model that always predicts "legitimate" would already score ~99.6% accuracy. This project instead evaluates using:
- **Macro AUC-ROC** and **macro F1-score** (equal weight across all classes, including rare ones)
- **Per-class precision-recall curves**, since PR curves are more informative than ROC under heavy imbalance
- **McNemar's test** for statistically comparing model performance

**Results (XGBoost, best-performing model):**

## Key Finding: Ablation Study
Removing the engineered velocity features (expanding mean/std, transaction count, category/merchant diversity, implied travel speed) **slightly improved** both macro AUC and macro F1, rather than degrading them. This suggests the velocity features were largely **redundant** with simpler signals already available to the model (transaction amount, hour, geographic distance) — a useful negative result that shaped how the features are described here, rather than being reported as unqualified improvements.

## Known Limitations
- **Rarest fraud class underrepresented in evaluation:** with only ~1,035 total fraud transactions spread across 10 fraud sub-types, the smallest class can have too few samples to reliably appear in every stratified test split — a known constraint of this dataset's size at this level of label granularity, not a code defect.
- **Partial taxonomy circularity risk:** a subset of the 11 fraud classes are rule-derived from variables (e.g., transaction hour, distance percentile) that are also used as model input features. This likely contributes to the very high macro AUC and should be kept in mind when interpreting results — the model may be partly re-deriving label-construction rules rather than discovering independent fraud signal.
- **Random (not time-based) train/test split:** the current split is stratified-random rather than chronological, which is a common simplification but can be optimistic for temporal fraud patterns compared to a strict time-based split.

## Tech Stack
Python, XGBoost, LightGBM, Scikit-learn, imbalanced-learn, SHAP, Pandas, NumPy, Matplotlib, Seaborn, SciPy (statistical testing)

## Possible Extensions
- Move to a time-based (chronological) train/test split for a more realistic evaluation
- Redefine rule-derived fraud classes using signals independent of the model's input features, to reduce circularity risk
- Collect or oversample additional examples for the rarest fraud sub-type to stabilize per-class evaluation
