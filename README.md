# Tata Steel Machine Failure Prediction

This project focuses on predicting machine failures in an industrial manufacturing setup using sensor and operational data.
The objective is to identify machines at risk of failure so that preventive maintenance can be planned in advance.

The dataset represents real-world machine operating conditions such as temperature, rotational speed, torque, and tool wear,
along with different types of failure indicators.

---

## Problem Statement

Machine failures in large-scale manufacturing plants like steel production units can lead to high operational and financial losses.
The goal of this project is to build a classification model that can predict whether a machine is likely to fail,
with a higher emphasis on identifying failure cases accurately.

---

## Dataset Overview

- Source: Tata Steel Machine Failure Dataset
- Total Records: ~136,000
- Target Variable: `Machine failure` (0 = No Failure, 1 = Failure)

### Key Features
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]
- Machine type
- Specific failure indicators:
  - Tool Wear Failure
  - Heat Dissipation Failure
  - Power Failure
  - Overstrain Failure
  - Random Failures

The dataset does not contain missing or duplicate values.

---

## Exploratory Data Analysis (EDA)

- Machine failure cases are rare, indicating a strong class imbalance.
- High correlation observed between overall machine failure and specific failure indicators.
- Rotational speed and torque show strong negative correlation.
- Temperature-related features show stable distributions with minimal variance.
- Most failures are directly associated with one or more specific failure types.

---

## Feature Engineering & Preprocessing

- Renamed abbreviated failure columns to full descriptive names.
- Applied standard scaling on numerical features.
- Used One-Hot Encoding for machine type.
- Handled all preprocessing steps using a ColumnTransformer for pipeline integration.

---

## Model Building

Multiple classification models were trained and evaluated:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM
- Balanced Random Forest

Evaluation metrics used:
- Precision
- Recall
- F1-score

Due to class imbalance, accuracy alone was not considered sufficient.

---

## Handling Class Imbalance

Several techniques were applied and compared:

- Class weight adjustment
- SMOTE oversampling
- Random undersampling + oversampling
- Balanced Random Forest
- Custom scoring function prioritizing recall

The trade-off between recall and precision was carefully analyzed,
keeping in mind that missing a real machine failure is more costly than inspecting a healthy machine.

---

## Model Selection & Tuning

- RandomizedSearchCV was used to tune Balanced Random Forest and Random Forest models.
- Custom scoring was applied with higher weightage to recall.
- Cross-validation was performed to ensure stability across folds.

Final model selection prioritized **high recall** while maintaining reasonable precision.

---

## Final Model Performance (Test Data)

- Recall (Failure class): ~88–90%
- Precision (Failure class): Lower, due to imbalance trade-off
- The model successfully identifies most failure cases, even at the cost of some false positives.

This behavior is acceptable for preventive maintenance use cases.

---

## Feature Importance Insights

The most influential features for predicting machine failure:
- Rotational speed
- Torque
- Tool wear
- Heat dissipation failure
- Overstrain failure

Temperature-related features had relatively lower impact.

---

## Final Prediction on Unseen Data

- Test records: ~90,000
- Predicted failures: ~6,200 machines
- Non-failure predictions: ~84,700 machines

This output can be directly used to flag machines for inspection or maintenance.

---

## Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn, Plotly
- Scikit-learn
- XGBoost
- LightGBM
- Imbalanced-learn
- Jupyter Notebook

---

## Conclusion

This project demonstrates an end-to-end machine learning pipeline for an imbalanced classification problem.
Rather than maximizing accuracy, the focus is on building a reliable early-warning system
that helps reduce costly machine failures in industrial environments.
