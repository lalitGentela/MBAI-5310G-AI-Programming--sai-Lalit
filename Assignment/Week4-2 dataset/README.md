# SwiftShip Logistics — Delivery Delay Risk Prediction
### Business Plan & Decision Tree Model Analysis
**AI Assignment 4_2dataset**

---

## 1. Business Overview

SwiftShip Logistics is a mid-sized delivery and fulfillment company that provides shipping services for small businesses, retail chains, e-commerce sellers, and enterprise clients across Canada. The company handles shipments from regional warehouses and works with multiple delivery carriers. Its customers expect reliable delivery, accurate communication, and fast action when shipments are at risk.

| Business Element | Description |
|---|---|
| Company Name | SwiftShip Logistics |
| Business Area | Delivery and Fulfillment / Logistics |
| Main Service | Shipping services for businesses across Canada via regional warehouses and multiple carriers |
| ML Task | Binary classification using a Decision Tree model |
| Target Variable | Delivery_Delayed (1 = delayed, 0 = on time) |
| Prediction Goal | Predict whether a shipment is at high risk of being delayed before the delivery deadline |

---

## 2. Business Problem

SwiftShip Logistics is facing a service reliability problem. Some shipments arrive later than promised, especially during high warehouse load, poor weather, long-distance routes, peak seasons, or when carrier performance is weak. Delayed shipments reduce customer satisfaction, increase support tickets, and damage business relationships.

The business wants to identify shipments likely to be delayed before the delivery deadline. This allows operations managers to take preventive action instead of reacting after the customer complains.

- **Reactive problem:** Without prediction, staff only discover delays after they happen, when it is too late to intervene.
- **Operational cost:** Unplanned delays increase support workload, urgent re-routing costs, and compensation claims.
- **Customer trust:** Repeated delays erode long-term relationships with retail and enterprise clients.

---

## 3. Decision That Machine Learning Can Support

The Decision Tree model classifies each shipment as either high-risk or low-risk for delay. This supports the following operational decisions:

- If the model predicts a delay: staff may prioritise the shipment, contact the carrier, upgrade the service level, or notify the customer early.
- If the model predicts on-time delivery: the company continues with the normal fulfillment process.
- Which carriers, routes, or warehouses are contributing most to delays (feature importance).
- Which shipments need urgent human review before dispatch.

The prediction supports operational decision-making but does not replace human judgment.

---

## 4. Target Variable and Input Features

**Target Variable:**

| Variable | Type | Meaning |
|---|---|---|
| Delivery_Delayed | Binary target | 1 = shipment delayed; 0 = shipment delivered on time |

**Input Features:**

| Feature Group | Examples from Dataset |
|---|---|
| Customer and business context | Customer_Segment, Region, Product_Category, Order_Value_CAD |
| Shipping information | Shipping_Method, Shipping_Distance_km, Package_Weight_kg, Promised_Delivery_Days |
| Operational conditions | Warehouse_Load_Level, Carrier_Rating, Peak_Season |
| Risk indicators | Weather_Risk, Previous_Delivery_Issues |

---

## 5. Notebook Workflow

### 5.1 Data Loading & Inspection

The dataset is loaded from an Excel file using pandas. Initial inspection uses `df.head()`, `df.tail()`, `df.shape`, `df.columns.tolist()`, and `df.info()` to understand structure and data types. Missing values are counted with `df.isnull().sum()` and duplicates with `df.duplicated().sum()`.

### 5.2 Feature Type Identification

Features are divided into numerical (`int64`, `float64`) and categorical (`object`) types. The target column `Delivery_Delayed` is identified separately. A bar chart is produced using `matplotlib` to visualise the count of numerical vs categorical features.

### 5.3 Data Cleaning

A copy of the original DataFrame (`df_clean`) is created to preserve raw data. Cleaning involves two steps:

- Duplicate rows are removed using `drop_duplicates()`.
- Missing numerical values are filled with the column median; missing categorical values are filled with the mode (most frequent value).

### 5.4 Feature & Target Separation

`X` is created by dropping `Delivery_Delayed` from `df_clean`. `y` is the target column. Both shapes are confirmed and the first five rows of each are displayed.

### 5.5 Train-Test Split

The data is split 80/20 using `train_test_split()` with `stratify=y` to preserve class distribution and `random_state=42` for reproducibility. Training and testing sizes are visualised with a bar chart.

### 5.6 Preprocessing

Categorical features are encoded using `OneHotEncoder(handle_unknown="ignore", sparse_output=False)`. Numerical features are passed through unchanged using a `ColumnTransformer`. Processed DataFrames (`x_train_processed_df` and `x_test_processed_df`) are created using the new feature names.

### 5.7 Decision Tree Training

A `DecisionTreeClassifier` with `max_depth=4` and `random_state=42` is trained on the processed training data. The tree is visualised using `plot_tree()`. Tree depth and number of leaf nodes are printed.

### 5.8 Predictions & Evaluation

Predictions (`y_pred_dt`) are generated on the test set. Actual vs predicted values are displayed. The following metrics are computed:

| Metric | Code Used | Business Relevance for SwiftShip |
|---|---|---|
| Accuracy | `accuracy_score(y_test, y_pred_dt)` | Overall proportion of correct delay/on-time predictions |
| Precision | `precision_score(..., pos_label=1)` | When model predicts a delay, how often it is correct |
| Recall | `recall_score(..., pos_label=1)` | How many truly delayed shipments are caught — most critical metric |
| F1-Score | `f1_score(..., pos_label=1)` | Balanced measure of precision and recall |
| Confusion Matrix | `confusion_matrix(y_test, y_pred_dt, labels=[0,1])` | TN, FP, FN, TP breakdown |
| Classification Report | `classification_report(y_test, y_pred_dt)` | Full per-class performance summary |

### 5.9 Overfitting Analysis

Two models are compared: the original controlled tree (`max_depth=4`) and an overfitted tree (`max_depth=None`). A second controlled tree is also built with `max_depth=4, min_samples_leaf=5`. Training and testing accuracy are calculated and plotted for each.

| Model | max_depth | min_samples_leaf | Expected Behaviour |
|---|---|---|---|
| Original Decision Tree | 4 | Default (1) | Balanced performance |
| Overfitted Decision Tree | None (unlimited) | Default (1) | Very high training accuracy, lower test accuracy |
| Controlled Decision Tree | 4 | 5 | Better generalisation, reduced overfitting risk |

Tree complexity (depth and number of leaves) is compared across all models in a final summary table.

---

## 6. Business Interpretation of Model Results

| Model Result | Business Interpretation for SwiftShip Logistics |
|---|---|
| High Accuracy | The model correctly classifies most shipments, but accuracy alone is not enough — recall must also be reviewed. |
| False Positive (FP) | The model predicts a delay, but the shipment arrives on time. Staff may spend extra resources on a shipment that did not need intervention. |
| False Negative (FN) | The model predicts on-time delivery, but the shipment is delayed. The company fails to act early, resulting in customer dissatisfaction and urgent support work. |
| High Precision | The model is selective about raising delay flags. Fewer unnecessary interventions, but some delayed shipments may be missed. |
| High Recall | The model catches most delayed shipments. Best outcome for customer experience, even if some false alarms occur. Recall is the most important metric for this business. |
| Feature Importance | Shows which variables most influence delay predictions — e.g. Warehouse_Load_Level, Weather_Risk, Carrier_Rating, Shipping_Distance_km, Previous_Delivery_Issues. |
| Overfitting Gap | A large gap between training and testing accuracy means the model has memorised the training data and will not generalise to new shipments. |

---

## 7. False Positives and False Negatives

| Error Type | Meaning | Business Consequence |
|---|---|---|
| False Positive | Model predicts delay, but shipment arrives on time | Extra staff time and resources spent on a shipment that did not need intervention |
| False Negative | Model predicts on time, but shipment is delayed | Company fails to act early — customer dissatisfaction, urgent support, reputational damage |

For SwiftShip Logistics, **false negatives are more costly** than false positives. Missing a delayed shipment has a direct impact on the customer experience and business relationships, making recall the priority metric.

---

## 8. Feature Importance: Business Use

The Decision Tree model ranks features by how much they influence the delay prediction. Key features and their business actions:

- **Warehouse_Load_Level:** If important, managers may adjust staffing or fulfillment scheduling during peak periods.
- **Carrier_Rating:** If important, the company may review carrier contracts or assign reliable carriers to high-value customers.
- **Weather_Risk:** If important, the company may send early customer warnings or use alternate routes.
- **Shipping_Distance_km:** If important, long-distance routes may need earlier dispatch windows or upgraded service levels.
- **Previous_Delivery_Issues:** If important, the company may monitor repeated problem routes or customer accounts more closely.

---

## 9. Limitations and Responsible AI Reflection

- The dataset is synthetic and simplified. Real-world delivery performance is influenced by traffic conditions, driver availability, address quality, warehouse system outages, and sudden carrier disruptions not captured here.
- The model may reflect past operational patterns, which could create bias against certain regions, routes, or customer groups if historical data is not reviewed carefully.
- A Decision Tree with unlimited depth (`max_depth=None`) will overfit, as demonstrated in the notebook. Controlling `max_depth` and `min_samples_leaf` reduces this risk.
- Human judgment is still necessary. Operations managers understand context — relationships, exceptions, real-time conditions — that does not appear in the dataset.

---

## 10. Conclusion

SwiftShip Logistics' Decision Tree model provides an interpretable, practical approach to identifying high-risk shipments before they are delayed. The notebook covers a complete pipeline: data loading and inspection, cleaning, feature/target separation, preprocessing with `OneHotEncoder`, train-test split, model training, evaluation, and overfitting analysis.

The controlled Decision Tree (`max_depth=4`, `min_samples_leaf=5`) balances predictive performance with generalisation. Given the business context, **recall is the most critical metric** — missing a delayed shipment has a greater business cost than a false alarm. Feature importance results guide operational actions around carriers, warehouses, routes, and weather risk.

The model should be used as a decision-support tool. Final intervention decisions remain the responsibility of operations managers.

---

## Repository Structure

    Week4-2 dataset/
    ├── AI_Assignment_4_2dataset.ipynb
    ├──swiftship_delivery_delay_dataset.xlsx
    ├── business_plan_swiftship_delivery_delay.pdf
    └── README.md
