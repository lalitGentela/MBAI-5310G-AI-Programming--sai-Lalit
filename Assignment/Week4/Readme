# SmartLoan Partners — Small Business Loan Pre-Approval System
### Business Plan & Decision Tree Model Analysis
**AI Assignment 4**

---

## 1. Business Overview

SmartLoan Partners is a financial technology company specialising in small and medium-sized business (SMB) lending. The company collects detailed application data from each borrower — including business age, annual revenue, monthly profit, credit score, existing debt, collateral value, cash-flow stability, and previous default history — and uses this data to power a machine learning-based pre-approval system.

The goal is to make the loan screening process faster, more consistent, and more data-driven, while retaining human oversight for final decisions.

| Business Element | Description |
|---|---|
| Company Name | SmartLoan Partners |
| Business Area | Financial Services / Small Business Lending |
| Main Service | Screening small business loan applications and supporting pre-approval decisions |
| ML Task | Binary classification using a Decision Tree model |
| Target Variable | Loan_Approved (Yes / No) |
| Prediction Goal | Predict whether a small business loan should be pre-approved or not |

---

## 2. Business Problem

Loan officers at SmartLoan Partners receive a high volume of applications every month. Reviewing each one manually is time-consuming and prone to inconsistency, as different officers may weigh the same risk factors differently. This creates two core problems:

- **Operational inefficiency:** Manual review delays approvals and increases staffing costs.
- **Inconsistent decisions:** Subjective judgement can lead to unfair or unpredictable outcomes for applicants.

Machine learning addresses both problems by generating an initial decision-support score for each application. This score does not replace loan officers but helps them prioritise high-confidence cases, flag borderline applications for closer review, and apply consistent criteria across all submissions.

---

## 3. Decision That Machine Learning Can Support

The Decision Tree model assists SmartLoan Partners in determining whether a loan application should be marked as likely approved or likely not approved. Specifically, it supports the following business decisions:

- Which applications should be reviewed first by loan officers.
- Which applications may need additional documentation or collateral.
- Which applicants may be too risky for automatic pre-approval.
- Which features are most predictive of approval (feature importance).

---

## 4. Dataset Description

The dataset contains synthetic small business loan application records. Each row represents one application. The dataset intentionally includes missing values and duplicate rows to allow practice with data cleaning.

| Variable Group | Examples | Business Meaning |
|---|---|---|
| Financial | Annual_Revenue, Monthly_Profit, Existing_Debt, Requested_Loan_Amount | Indicate financial strength and repayment capacity |
| Risk | Credit_Score, Previous_Default, Cash_Flow_Stability | Estimate likelihood of default or risky behaviour |
| Relationship | Relationship_Length_Months, Online_Application_Complete | Reflect existing business relationship and process completion |
| Business Profile | Industry, Region, Business_Age_Years | Describe the applicant's business context |
| Target Variable | Loan_Approved | Binary label: Yes (approved) or No (not approved) |

---

## 5. Notebook Workflow

### 5.1 Data Loading & Inspection

The dataset is loaded from an Excel file using pandas. Initial inspection includes `df.head()`, `df.tail()`, `df.shape`, `df.columns`, and `df.info()` to understand the structure and data types. Missing values are counted with `df.isnull().sum()` and duplicates with `df.duplicated().sum()`.

### 5.2 Data Cleaning

A copy of the original DataFrame (`df_clean`) is created to preserve the raw data. Cleaning involves two steps:

- Duplicate rows are removed using `drop_duplicates()`.
- Missing numerical values are filled with the column median; missing categorical values are filled with the mode (most frequent value).

### 5.3 Feature & Target Separation

Features are split into numerical (`int64`, `float64`) and categorical (`object`) types. The target column `Loan_Approved` is separated from the input features to create `X` (features) and `y` (target). The target class distribution is inspected to check for imbalance.

### 5.4 Train-Test Split

The data is split 80/20 using `train_test_split()` with `stratify=y` to preserve the class distribution across both sets and `random_state=42` for reproducibility.

### 5.5 Preprocessing

Categorical features are encoded and numerical features are scaled using a preprocessing pipeline. Processed DataFrames (`x_train_processed_df` and `x_test_processed_df`) are used for model training and evaluation.

### 5.6 Decision Tree Training & Evaluation

A Decision Tree classifier (`decision_tree_model`) is trained on the processed training data. Predictions (`y_pred_dt`) are generated on the test set.

| Metric | Code Used | Business Relevance |
|---|---|---|
| Accuracy | `accuracy_score(y_test, y_pred_dt)` | Overall proportion of correct predictions |
| Precision | `precision_score(..., pos_label='Yes')` | Avoid approving risky applicants (false positives) |
| Recall | `recall_score(..., pos_label='Yes')` | Avoid missing good applicants (false negatives) |
| F1-Score | `f1_score(..., pos_label='Yes')` | Balanced measure of precision and recall |
| Classification Report | `classification_report(...)` | Full per-class breakdown |
| Confusion Matrix | `confusion_matrix(y_test, y_pred_dt)` | TN, FP, FN, TP counts |

### 5.7 Overfitting Analysis

Two models are compared: the controlled Decision Tree (`max_depth=4`, `min_samples_leaf=5`) and an overfitted Decision Tree (`max_depth=None`). Both training and testing accuracy are calculated for each model.

| Model | max_depth | min_samples_leaf | Expected Behaviour |
|---|---|---|---|
| Overfitted Tree | None (unlimited) | Default (1) | Very high training accuracy, lower test accuracy |
| Controlled Tree | 4 | 5 | Slightly lower training accuracy, better generalisation |

---

## 6. Business Interpretation of Model Results

| Model Result | Business Interpretation |
|---|---|
| High Accuracy | The model correctly classifies most applications. However, accuracy alone is insufficient; precision and recall must also be reviewed. |
| False Positive (FP) | The model predicts approval, but the application should have been rejected. This may result in financial loss if a risky borrower receives a loan. |
| False Negative (FN) | The model predicts rejection, but the application could have been approved. This risks lost revenue and may unfairly exclude creditworthy applicants. |
| High Precision | The model is conservative in approvals, minimising risky loans being funded. |
| High Recall | The model avoids missing good applicants. Suitable when growing the loan book is the priority. |
| Feature Importance | Identifies which variables (e.g. Credit_Score, Monthly_Profit, Existing_Debt) most influence approval decisions. |
| Overfitting Gap | A large gap between training and testing accuracy signals the model has memorised training data and will not generalise to new applications. |

---

## 7. Limitations and Potential Bias

- The dataset is synthetic. Real loan decisions involve more complex financial records, legal requirements, and market conditions not captured here.
- The model may inherit historical bias if past approval decisions were unfair, or if certain industries or regions are over- or under-represented.
- A Decision Tree with unlimited depth can overfit, as demonstrated in the notebook. Controlling `max_depth` and `min_samples_leaf` mitigates this.
- The model cannot account for qualitative context, such as economic conditions, relationship history, or extenuating business circumstances.

---

## 8. Why Human Judgment Remains Essential

- Loan approval decisions affect real businesses and livelihoods. A model error carries significant human and financial consequences.
- Loan officers can assess documentation, context, and fairness in ways a Decision Tree cannot.
- Legal and regulatory obligations in financial services require accountable human decision-makers.
- The model is a decision-support tool, not a replacement for professional judgement.

---

## 9. Conclusion

SmartLoan Partners' Decision Tree model represents a practical and interpretable approach to automating the loan pre-screening process. The notebook demonstrates a complete machine learning pipeline: from data loading and cleaning, through feature engineering and model training, to rigorous evaluation and overfitting analysis.

The controlled Decision Tree (`max_depth=4`, `min_samples_leaf=5`) strikes an appropriate balance between predictive performance and generalisation. Evaluation metrics — particularly precision, recall, and F1-score — must be interpreted in the context of SmartLoan Partners' business priorities: minimising risky approvals while still capturing creditworthy applicants.

Ultimately, the model is most valuable as a tool that increases consistency and efficiency in the screening process, while loan officers retain the authority and responsibility for final decisions.

---

## 10. References

- SmartLoan Partners Teaching Case — Synthetic Dataset for AI Programming / Business Analytics
- Assignment 4 Jupyter Notebook — AI_Assignment_4.ipynb
- scikit-learn Documentation — DecisionTreeClassifier, train_test_split, evaluation metrics
- pandas Documentation — DataFrame operations, missing value handling, duplicate removal
