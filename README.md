Loan Approval Prediction using Machine Learning
Melbourne Institute of Technology – MDA512: Data Science

Date: July 2025

📘 Overview

This repository contains an end-to-end data-science project that predicts loan approval outcomes using multiple machine-learning algorithms.
The objective is to automate and enhance the efficiency, accuracy, and fairness of loan-approval decisions in banking institutions.

The project follows a complete ML pipeline — from data preprocessing and feature engineering to model training, evaluation, and business interpretation.

🧩 Problem Statement

Manual loan approvals are time-consuming, inconsistent, and prone to bias.
This project develops predictive models to classify applications as Approved (Y) or Rejected (N) using historical data from Kaggle’s Loan Prediction Dataset.

🎯 Objectives

Build supervised ML models to predict loan-approval status.

Identify key financial and demographic drivers of approval.

Evaluate the effect of income-to-loan ratio on loan decisions.

Compare performance across algorithms (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost).

Provide business insights linking model performance to profitability and risk management.

🧮 Dataset

Source: Kaggle – Loan Prediction Dataset

Records: 614

Features: 12 (mix of categorical + numerical)

Target: Loan_Status (Y/N)

Key Attributes:
Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area.

⚙️ Preprocessing & Feature Engineering

Data Cleaning: Removed identifiers, fixed “3+” dependents, imputed missing values (median/mode).

Feature Engineering: Created

Total_Income = Applicant + Coapplicant Income

EMI_Ratio = LoanAmount / Total_Income

Income_to_Loan = Total_Income / LoanAmount

Log-transformed skewed variables.

Encoding & Scaling: One-Hot Encoding + StandardScaler.

🧠 Models Implemented
Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	0.8699	0.8485	0.9882	0.9130	0.8706
Decision Tree	0.8455	0.8438	0.9529	0.8950	0.7396
Random Forest	0.8943	0.9091	0.9412	0.9249	0.8627
Gradient Boosting	0.8455	0.8587	0.9294	0.8927	0.8186
XGBoost	0.8293	0.8636	0.8941	0.8786	0.8220

✅ Best Model: Random Forest – Highest overall accuracy and F1 score.

📊 Visualisation Highlights

Distributions: Log-transformed income and loan values.

Credit History vs Approval: Applicants with credit history ≈ 80 % approval rate.

Income-to-Loan Trend: Moderate ratios (20 – 60) show maximum approval.

ROC Curves & Confusion Matrices: Model performance comparison.

🧩 Tools & Libraries

Language: Python 3.10+

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost

Environment: Jupyter Notebook / Google Colab

🏦 Business Impact

Faster Decisions: Automated approvals reduce processing time.

Consistency & Fairness: ML models minimise human bias.

Profitability: Early identification of high-risk borrowers reduces defaults.

Customer Experience: Quicker turnaround boosts trust and satisfaction.

🚀 Future Work

Integrate external credit-bureau or macroeconomic data.

Address class imbalance with SMOTE or weighted loss functions.

Expand dataset → test deep-learning models (e.g., ANN).

Build a Flask or FastAPI web dashboard for real-time loan prediction.

📂 Repository Structure
Loan-Approval-Prediction/
│
├── data/                     # Raw and cleaned datasets
├── notebooks/                # Jupyter or HTML analysis files
│   ├── MDA512_Assignment2_Samar_grp1.html
│   └── MDA512_Assignment2_Samar_grp1.pdf
├── src/                      # (optional) Python scripts for preprocessing and model training
├── README.md                 # Project overview (this file)
└── requirements.txt          # Python dependencies

🧾 Citation

Kaggle Dataset – Ninzaami, “Loan Prediction Dataset,” 2024. [Online]. Available: https://www.kaggle.com/datasets/ninzaami/loan-predication

👥 Authors

Melbourne Samar Group 1

Supervised by: MDA512 Teaching Team, Melbourne Institute of Technology
