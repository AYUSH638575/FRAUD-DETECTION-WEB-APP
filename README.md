# Fraud Detection Prediction App

A machine learning project that detects fraudulent financial transactions using Logistic Regression, with an interactive Streamlit app for real-time predictions.

## Overview

This project analyzes a large dataset of mobile money transactions to identify patterns of fraud, then trains a classification pipeline to predict whether a given transaction is fraudulent. The trained model is deployed through a simple Streamlit web app where users can input transaction details and get an instant prediction.

## Dataset

The dataset (`AIML Dataset.csv`, not included in this repo due to size) contains transaction-level financial data with the following key fields:

- `step` — time unit of the transaction
- `type` — transaction type (`PAYMENT`, `TRANSFER`, `CASH_OUT`, `DEPOSIT`)
- `amount` — transaction amount
- `nameOrig`, `oldbalanceOrg`, `newbalanceOrig` — sender ID and balances before/after
- `nameDest`, `oldbalanceDest`, `newbalanceDest` — receiver ID and balances before/after
- `isFraud` — target label (1 = fraud, 0 = not fraud)
- `isFlaggedFraud` — flag raised by the existing rule-based system

The dataset is highly imbalanced, with fraudulent transactions making up only **0.13%** of all records (8,213 fraud cases out of ~6.36 million transactions).

## Exploratory Data Analysis

The analysis notebook (`analysis_model.ipynb`) covers:

- Class distribution of `isFraud` and `isFlaggedFraud`
- Transaction type frequency and fraud rate by type
- Distribution of transaction amounts (log scale)
- Amount vs. fraud outcome comparisons
- Engineered features: `balanceDiffOrig` and `balanceDiffDest` (change in sender/receiver balance)
- Fraud trends over time
- Top senders/receivers and repeat fraud users
- Fraud concentration within `TRANSFER` and `CASH_OUT` transaction types
- Correlation analysis between numeric features and the fraud label

## Model

**Pipeline:**
- Preprocessing via `ColumnTransformer`:
  - `StandardScaler` on numeric features (`amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`)
  - `OneHotEncoder` on the categorical `type` feature
- Classifier: `LogisticRegression` with `class_weight="balanced"` to handle class imbalance

**Train/test split:** 70/30, stratified on the target label.

### Results

```
              precision    recall  f1-score   support

           0       1.00      0.95      0.97   1906322
           1       0.02      0.94      0.04      2464

    accuracy                           0.95   1908786
```

The model achieves **94% recall** on fraud cases, meaning it catches the large majority of fraudulent transactions. Precision on the fraud class is low (2%), which is an expected tradeoff of `class_weight="balanced"` on such a heavily imbalanced dataset — the model flags many legitimate transactions as suspicious in order to minimize missed fraud. This favors a "cast a wide net, review flagged cases" use case over fully automated blocking.

The trained pipeline is saved as `fraud_detection_pipeline.pkl`.

## Streamlit App

`fraud_detection.py` loads the trained pipeline and provides a simple UI to:

1. Select the transaction type
2. Enter the transaction amount
3. Enter sender's old/new balance
4. Enter receiver's old/new balance
5. Click **Predict** to see whether the transaction is likely fraudulent

## Project Structure

```
.
├── analysis_model.ipynb           # EDA + model training notebook
├── fraud_detection_pipeline.pkl   # Trained model pipeline
├── fraud_detection.py             # Streamlit app
└── README.md
```

## Setup & Usage

### Requirements

```bash
pip install streamlit pandas scikit-learn joblib matplotlib seaborn
```

### Run the app

```bash
streamlit run fraud_detection.py
```

Make sure `fraud_detection_pipeline.pkl` is in the same directory as `fraud_detection.py`.

### Reproducing the model

Open `analysis_model.ipynb` and run all cells (requires `AIML Dataset.csv` in the working directory). The final cells retrain the pipeline and export it to `fraud_detection_pipeline.pkl`.

## Future Improvements

- Address low precision with alternative approaches (e.g., threshold tuning, SMOTE/undersampling, tree-based models like Random Forest or XGBoost)
- Add feature importance / SHAP explanations to the app
- Incorporate engineered features (`balanceDiffOrig`, `balanceDiffDest`) into the model pipeline
- Add model evaluation metrics (ROC-AUC, PR-AUC) given the class imbalance
- Deploy the app (e.g., Streamlit Community Cloud)

## Tech Stack

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Streamlit
- joblib
