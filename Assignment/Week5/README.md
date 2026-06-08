# FitLife Studio Analytics — Membership Renewal Risk Prediction
### Business Plan & K-Means Clustering Analysis
**AI Assignment 5**

---

## 1. Business Overview

FitLife Studio is a mid-sized fitness studio chain that sells monthly, six-month, and annual memberships. The business offers gym access, group classes, personal training, and a mobile app that helps members book classes and track visits. The company earns most of its revenue from recurring membership renewals, so keeping existing members is a major business priority.

The studio operates in a competitive market where customers can easily switch to another gym, home workout app, or lower-cost fitness option. FitLife wants to use data to discover hidden customer segments and identify early signs that a member may not renew.

| Business Element | Description |
|---|---|
| Company Name | FitLife Studio Analytics |
| Business Area | Fitness subscription services and customer retention |
| Business Decision | Identify members who are at risk of not renewing their membership |
| ML Task | Unsupervised learning using K-Means Clustering |
| Target Variable | No target variable — customer segments are discovered from features |
| Goal | Group members into meaningful segments to support targeted retention actions |

---

## 2. Business Problem

The main problem is membership churn. Some members stop attending classes, log into the app less often, report lower satisfaction, or experience payment and service problems before they decide not to renew. If the business waits until the membership expires, it may be too late to recover the customer relationship.

| Business Challenge | Why It Matters | Possible Business Action |
|---|---|---|
| Low member engagement | Members who rarely visit may feel less value from the membership | Send personalised class suggestions or check-in messages |
| Low satisfaction or complaints | Dissatisfied members are more likely to leave | Assign service follow-up or offer a recovery plan |
| Payment issues | Payment problems can create friction and cancellation risk | Offer billing support or flexible payment options |
| High distance from studio | Convenience affects renewal decisions | Recommend online classes or closer branch options |
| Monthly contract type | Monthly members can leave more easily than annual members | Offer renewal incentives or longer contract benefits |

---

## 3. Why Unsupervised Learning (K-Means Clustering)?

Unlike supervised learning, this assignment does not use a labelled target variable (Yes/No) to train a model. Instead, K-Means Clustering is used to discover hidden patterns in customer behaviour without predefined labels.

| Concept | Supervised Learning | Unsupervised Learning (This Assignment) |
|---|---|---|
| Labels | X = features, y = target | X = features only, no y |
| Goal | Predict a known outcome | Discover hidden groups |
| Algorithm | Decision Tree | K-Means Clustering |
| Output | Class prediction | Cluster assignments |

This approach allows FitLife to discover which natural customer segments exist in the data, then design targeted retention strategies for each segment.

---

## 4. Dataset & Features Selected

The dataset contains synthetic FitLife member records. Each row represents one member. The following 10 numerical features were selected for clustering:

| Feature | Business Meaning |
|---|---|
| Age | Member's age — may relate to engagement patterns and preferences |
| Monthly_Fee | Amount paid per month — indicates membership tier and value perception |
| Months_As_Member | Tenure — longer members may be more loyal |
| Avg_Weekly_Visits | Average weekly gym visits — key engagement indicator |
| Classes_Per_Month | Number of classes attended — reflects activity level |
| App_Login_Days_Last_30 | App engagement in the last 30 days — digital engagement signal |
| Payment_Issues_Last_6_Months | Count of payment problems — renewal friction indicator |
| Satisfaction_Score | Member satisfaction rating — key churn predictor |
| Complaint_Count_Last_3_Months | Number of complaints — service dissatisfaction indicator |
| Distance_From_Studio_km | Distance from nearest studio — convenience factor |

**Note:** Categorical columns (Gender, Membership_Type, Contract_Type, etc.) were excluded from clustering. Only numerical columns were used after median imputation for missing values.

---

## 5. Notebook Workflow

### 5.1 Data Loading & Inspection

The dataset is loaded from an Excel file using pandas. Initial inspection uses `df.shape`, `df.columns`, and `df.info()` to understand the structure and data types.

### 5.2 Feature Selection

19 features are initially identified. For clustering, 10 numerical columns are selected into `x`. The notebook explicitly explains that this is unsupervised learning — there is no `y` target variable.

### 5.3 Data Cleaning

A copy of the dataset (`df_clean`) is created. Column names are stripped of extra spaces using `str.strip()`. Duplicate rows are identified and removed using `drop_duplicates()`. Missing values in the 10 numerical columns are filled with the column median using `fillna(median)`.

### 5.4 Feature Scaling

`StandardScaler` from scikit-learn is applied to standardise all numerical features to a mean of 0 and standard deviation of 1. This is critical for K-Means because the algorithm uses distance — unscaled features with larger ranges would dominate the clustering.

```python
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
```

### 5.5 Elbow Method — Choosing the Best K

The Elbow Method is used to find the optimal number of clusters. `KMeans` is trained for K values from 1 to 10, and the inertia (sum of squared distances to cluster centroids) is recorded for each K.

```python
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(x_scaled)
    inertia.append(kmeans.inertia_)
```

**Elbow Analysis:** The inertia decreases sharply from K=1 to K=4. From K=4 onwards, the curve flattens gradually with no sharp drop. This means K=4 is where the rate of improvement meaningfully slows down — the elbow point is at **K=4**.

### 5.6 K-Means Model Training

The final K-Means model is trained with `n_clusters=3`, `random_state=42`, and `n_init=10`. Cluster labels are assigned to each member using `fit_predict()`.

```python
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(x_scaled)
```

### 5.7 Cluster Analysis & Segment Naming

Cluster labels are added to `df_clustered`. A cluster summary is produced using `groupby("Cluster").mean()` to calculate the average feature value per cluster. Segments are named based on the highest `Months_As_Member` (Active High Spenders), highest `Monthly_Fee` (High-Income Premium Buyers), and the remaining group (Low-Engagement Customers).

### 5.8 Visualisation

Two scatter plots are produced using `matplotlib`:
- **Plot 1:** Monthly_Fee vs Months_As_Member, coloured by cluster number.
- **Plot 2:** Same axes, coloured by segment name with a legend for business readability.

---

## 6. Customer Segments Discovered

| Cluster | Segment Name | Business Interpretation | Possible Business Strategy |
|---|---|---|---|
| 0 | Loyal Engaged Members | High visits, high app use, high satisfaction, long tenure | Loyalty rewards, early renewal incentives, referral bonuses, exclusive member events |
| 1 | Premium Dissatisfied Members | Pay the highest fees but visit least and are least satisfied | Proactive outreach, service quality review, personalised check-ins, fee-value alignment |
| 2 | Budget Low-Activity Members | Largest group, low engagement, moderate satisfaction, lowest fees | Class variety promotions, gamified attendance challenges, upgrade incentives |
| 3 | At-Risk Complaint-Prone Members | Newest members, dramatically higher complaint count than all other groups | Immediate complaint resolution pipeline, personal trainer introductions, satisfaction follow-ups |

---

## 7. Business Interpretation of Cluster Results

| Cluster Finding | Business Interpretation for FitLife |
|---|---|
| Loyal Engaged Members identified | These members are low churn risk. Focus retention resources on other segments. Reward their loyalty to encourage referrals. |
| Premium Dissatisfied Members identified | High-value members who are disengaged. Missing this group causes the greatest revenue loss per member. Prioritise service recovery. |
| Budget Low-Activity Members identified | Largest segment. Low engagement may signal low perceived value. Targeted class promotions could increase visits and reduce churn. |
| At-Risk Complaint-Prone Members identified | Newest members with high complaints — critical early intervention window. Unresolved complaints at this stage lead to rapid churn. |
| Elbow at K=4 | Four distinct customer types exist in the data, confirming that a one-size-fits-all retention approach is insufficient. |

---

## 8. Limitations and Responsible AI Reflection

- The dataset is synthetic and may not capture all reasons why a person renews or cancels. Real decisions may be influenced by health, income, schedule changes, family responsibilities, or transportation.
- Only 10 numerical features were used for clustering. Categorical features (Gender, Membership_Type, Contract_Type, etc.) were excluded, which may limit the completeness of the segments.
- K-Means assumes clusters are spherical and equal in size, which may not reflect real customer behaviour patterns.
- There is a risk of bias if certain membership types or locations are overrepresented in one cluster, leading to different treatment of those groups.
- The segment names (e.g. "At-Risk Complaint-Prone") are interpretations assigned after clustering — they should be reviewed by business managers before being used for customer communication.
- **Responsible AI:** Human judgment should still be used. The model cannot fully understand customer context, fairness concerns, or long-term brand relationships.

---

## 9. Conclusion

FitLife Studio's K-Means Clustering model provides a data-driven approach to understanding customer segments and identifying members at risk of not renewing. The notebook covers a complete unsupervised learning pipeline: data loading, inspection, cleaning, feature scaling with `StandardScaler`, Elbow Method analysis, K-Means model training with `n_clusters=3`, cluster analysis, segment naming, and visualisation.

The Elbow Method identified K=4 as the optimal number of clusters. Four meaningful customer segments were discovered are Loyal Engaged Members, Premium Dissatisfied Members, Budget Low-Activity Members, and At-Risk Complaint-Prone Members — each requiring a different retention strategy.

Unlike supervised learning, this approach does not predict a known label. Instead, it helps FitLife discover hidden structure in member behaviour that can guide proactive, personalised retention decisions before the renewal deadline.

---

## 10. References

- FitLife Studio Analytics Teaching Case — Synthetic Dataset for AI Programming / Business Analytics
- Assignment 5 Jupyter Notebook — AI_Assignment5.ipynb
- scikit-learn Documentation — KMeans, StandardScaler, fit_predict
- pandas Documentation — DataFrame operations, groupby, missing value handling
- matplotlib Documentation — scatter plots, Elbow Method visualisation
