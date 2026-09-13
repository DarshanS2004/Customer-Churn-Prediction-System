# Customer Churn Prediction System

An end-to-end machine learning application designed to analyze customer behaviour and predict the likelihood of customer churn.

The project combines data preprocessing, exploratory data analysis, feature engineering, class-imbalance handling, multiple machine learning algorithms, model evaluation, and an interactive Streamlit dashboard into a complete customer churn prediction workflow.

---

## Overview

Customer churn is a critical business problem in which customers stop engaging with a product or service.

This project uses historical customer and transaction-related information to identify patterns associated with churn and build classification models capable of predicting whether a customer is likely to churn.

The solution provides two complementary capabilities:

- **Business Analytics** — understand customer behaviour and churn patterns.
- **Churn Prediction** — evaluate machine learning models and predict churn for new customer inputs.

---

## Key Capabilities

### Data Preparation

- Missing-value treatment
- Duplicate detection and removal
- Categorical data normalization
- Date conversion
- Feature engineering
- Outlier handling using the IQR method
- Categorical encoding
- Numerical feature scaling

### Exploratory Analysis

The project analyzes churn relationships across:

- Age
- Income
- Spending Score
- Purchase Amount
- Returns
- Review Score
- Session Time
- Days Since Last Purchase
- Gender
- Product Category
- Payment Method
- City
- Country
- Device
- Browser

### Machine Learning

Multiple classification algorithms are trained and compared:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Support Vector Classifier
- Gradient Boosting
- Naive Bayes
- XGBoost

### Model Evaluation

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

### Interactive Dashboard

The Streamlit application provides:

- Customer churn overview
- Business KPIs
- Churn visualizations
- Model performance comparison
- Best-performing model identification
- Interactive churn prediction
- Churn probability when supported by the model

---

## Machine Learning Workflow

```text
Customer Dataset
       │
       ▼
Data Exploration
       │
       ▼
Data Cleaning
       │
       ├── Missing Values
       ├── Duplicate Removal
       └── Data Normalization
       │
       ▼
Feature Engineering
       │
       ▼
Outlier Handling
       │
       ▼
Categorical Encoding
       │
       ▼
Feature Scaling
       │
       ▼
Train / Test Split
       │
       ▼
SMOTE
       │
       ▼
Model Training
       │
       ├── Logistic Regression
       ├── KNN
       ├── Decision Tree
       ├── Random Forest
       ├── SVC
       ├── Gradient Boosting
       ├── Naive Bayes
       └── XGBoost
       │
       ▼
Model Evaluation
       │
       ▼
Model Selection
       │
       ▼
Saved Churn Model
       │
       ▼
Streamlit Prediction Dashboard
```

---

# Dataset

The project uses the following dataset:

```text
churn prediction.csv
```

The dataset contains customer-related attributes used to analyze behavioural patterns and predict churn. :contentReference[oaicite:3]{index=3}

---

# Data Preprocessing

Data preparation is performed before training the machine learning models.

## Missing Values

Categorical columns are filled using their mode, while numerical columns are filled using their median.

```python
categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

numerical_cols = df.select_dtypes(include=np.number).columns

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())
```

This ensures that missing values do not prevent the downstream machine learning workflow from operating. :contentReference[oaicite:4]{index=4}

---

## Duplicate Removal

Duplicate records are identified and removed from the dataset.

```python
df.drop_duplicates(inplace=True)
```

The preprocessing workflow identified 200 duplicate records before removal. :contentReference[oaicite:5]{index=5}

---

## Data Normalization

Inconsistent categorical values are normalized.

For example, gender values are standardized:

```text
M      → Male
F      → Female
male   → Male
female → Female
```

Country values are also normalized, including variations such as `IND`, `IN`, and `india`. :contentReference[oaicite:6]{index=6}

---

# Feature Engineering

The project converts `LastPurchaseDate` into a datetime value and creates a derived feature:

```text
DaysSinceLastPurchase
```

This represents the number of days since the customer's most recent purchase. :contentReference[oaicite:7]{index=7}

This feature provides additional behavioural information that can be used when analyzing customer churn.

---

# Exploratory Data Analysis

The project performs exploratory analysis to understand relationships between customer characteristics and churn.

## Numerical Analysis

The following numerical attributes are analyzed:

```text
Age
Income
SpendingScore
PurchaseAmount
Returns
ReviewScore
SessionTime
DaysSinceLastPurchase
```

Histograms and box plots are used to understand distributions and their relationship with churn. :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}

---

## Categorical Analysis

The project analyzes churn across categorical variables including:

```text
Gender
ProductCategory
PaymentMethod
City
Country
Device
Browser
```

Count plots are used to visualize how churn varies across these categories. :contentReference[oaicite:10]{index=10}

---

## Correlation Analysis

A correlation matrix is generated for numerical features to help understand relationships between variables.

```text
Numerical Features
        │
        ▼
Correlation Matrix
        │
        ▼
Heatmap Visualization
```

:contentReference[oaicite:11]{index=11}

---

# Outlier Handling

Potential numerical outliers are identified using box plots.

The project applies the **Interquartile Range (IQR)** method:

```text
Q1 = 25th Percentile
Q3 = 75th Percentile
IQR = Q3 - Q1

Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Records outside the calculated bounds are removed during preprocessing. :contentReference[oaicite:12]{index=12}

---

# Feature Encoding

Categorical features are converted into machine-learning-compatible numerical representations.

## Label Encoding

Label encoding is applied to selected categorical columns such as:

```text
Gender
Country
Churn
```

## One-Hot Encoding

One-hot encoding is applied to:

```text
ProductCategory
PaymentMethod
City
Device
Browser
```

with the first category dropped to avoid redundant dummy variables. :contentReference[oaicite:13]{index=13}

---

# Feature Scaling

Numerical features are standardized using `StandardScaler`.

The scaled features include:

```text
Age
Income
SpendingScore
PurchaseAmount
Returns
ReviewScore
SessionTime
DaysSinceLastPurchase
```

:contentReference[oaicite:14]{index=14}

---

# Handling Class Imbalance

The project uses **SMOTE (Synthetic Minority Over-sampling Technique)** on the training data.

```python
smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)
```

SMOTE is applied after the train/test split so that synthetic samples are generated only for the training data. :contentReference[oaicite:15]{index=15}

---

# Classification Models

The project trains and evaluates multiple supervised learning algorithms.

## 1. Logistic Regression

```python
LogisticRegression(max_iter=1000)
```

Used as a linear classification baseline. :contentReference[oaicite:16]{index=16}

## 2. K-Nearest Neighbors

```python
KNeighborsClassifier()
```

Used to classify customers based on neighbouring observations. :contentReference[oaicite:17]{index=17}

## 3. Decision Tree

```python
DecisionTreeClassifier(random_state=42)
```

Provides a tree-based classification approach. :contentReference[oaicite:18]{index=18}

## 4. Random Forest

```python
RandomForestClassifier(random_state=42)
```

Provides an ensemble-based classification approach using multiple decision trees. :contentReference[oaicite:19]{index=19}

## 5. Support Vector Classifier

```python
SVC()
```

Used to classify customers using a support-vector-based decision boundary. :contentReference[oaicite:20]{index=20}

## 6. Gradient Boosting

```python
GradientBoostingClassifier()
```

Used as a boosting-based classification model. :contentReference[oaicite:21]{index=21}

## 7. XGBoost

```python
XGBClassifier(eval_metric="logloss")
```

Used as an additional gradient-boosted tree classification model. :contentReference[oaicite:22]{index=22}

## 8. Gaussian Naive Bayes

```python
GaussianNB()
```

Provides a probabilistic classification approach. :contentReference[oaicite:23]{index=23}

---

# Model Evaluation

Each trained model is evaluated using several classification metrics.

| Metric | Purpose |
|---|---|
| Accuracy | Overall prediction correctness |
| Precision | Correct positive predictions among predicted positives |
| Recall | Positive cases correctly identified |
| F1 Score | Balance between precision and recall |
| ROC-AUC | Classification ranking performance |
| Confusion Matrix | Detailed prediction breakdown |

The project evaluates all trained models using a common evaluation function. :contentReference[oaicite:24]{index=24}

---

# Model Comparison

Evaluation results are collected into a DataFrame containing:

```text
Model
Accuracy
Precision
Recall
F1 Score
ROC-AUC
```

The models are then sorted by accuracy to compare their performance. :contentReference[oaicite:25]{index=25}

The Streamlit dashboard presents this comparison interactively and identifies the highest-accuracy model in the evaluation results. :contentReference[oaicite:26]{index=26}

---

# Model Persistence

The churn model is saved as:

```text
churn_model.pkl
```

The feature-column information is saved as:

```text
columns.pkl
```

The Streamlit application loads both files when it starts. :contentReference[oaicite:27]{index=27} :contentReference[oaicite:28]{index=28}

---

# Streamlit Dashboard

The project includes an interactive Streamlit application with three main sections:

```text
Customer Churn Dashboard
│
├── Dashboard
├── Model Performance
└── Prediction
```

:contentReference[oaicite:29]{index=29}

---

## Dashboard

The dashboard provides key business indicators including:

- Total Customers
- Churn Rate
- Average Income
- Average Spending Score

:contentReference[oaicite:30]{index=30}

It also visualizes:

- Churn Distribution
- Income vs Churn
- Spending Score vs Churn
- Session Time vs Churn

:contentReference[oaicite:31]{index=31}

---

# Model Performance

The Model Performance page displays the evaluation results of the trained classification models.

It includes:

- Model comparison table
- Accuracy comparison chart
- Best-performing model

:contentReference[oaicite:32]{index=32}

---

# Churn Prediction

The Prediction page allows users to enter customer information and generate a churn prediction.

The current interface accepts:

```text
Age
Income
Spending Score
Purchase Amount
Returns
Session Time
```

:contentReference[oaicite:33]{index=33}

The input values are transformed into the feature structure expected by the saved model before prediction. :contentReference[oaicite:34]{index=34}

---

## Prediction Output

The application displays one of two outcomes:

```text
⚠ Customer will churn
```

or

```text
✅ Customer will not churn
```

When the loaded model supports probability prediction, the application also displays the estimated churn probability. :contentReference[oaicite:35]{index=35}

---

# Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning |
| Imbalanced-learn | SMOTE |
| XGBoost | Gradient boosting |
| Pickle | Model persistence |
| Streamlit | Interactive dashboard |

---

# Project Structure

```text
Customer-Churn-Prediction-System/
│
├── app.py
├── churn prediction.csv
│
├── churn_model.pkl
├── columns.pkl
│
├── requirements.txt
├── README.md
└── .gitignore
```

> Update the structure if your actual project contains additional notebooks, datasets, scripts, or supporting files.

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/DarshanS2004/Customer-Churn-Prediction-System.git
```

## Navigate to the Project

```bash
cd Customer-Churn-Prediction-System
```

## Create a Virtual Environment

```bash
python -m venv .venv
```

## Activate the Environment

### Windows

```powershell
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

The application will be available at the local Streamlit address displayed in the terminal.

Typically:

```text
http://localhost:8501
```

---

# Example Prediction Workflow

```text
Enter Customer Information
          │
          ▼
Prepare Input Features
          │
          ▼
Align With Training Columns
          │
          ▼
Load Saved Churn Model
          │
          ▼
Generate Prediction
          │
          ▼
Display Churn Status
          │
          ▼
Display Probability
(if supported)
```

---

# Business Applications

A churn prediction system can support business teams in identifying customers who may require additional engagement.

Potential applications include:

- Customer retention analysis
- Churn-risk identification
- Customer engagement strategies
- Retention campaign targeting
- Behavioural analysis
- Customer segmentation support
- Business intelligence reporting

The predictions should be treated as analytical signals rather than guaranteed outcomes.

---

# Project Highlights

### End-to-End Machine Learning Workflow

The project covers the complete process from raw customer data to deployed predictions.

### Multiple Model Comparison

Eight classification algorithms are trained and evaluated using consistent performance metrics.

### Imbalanced Data Handling

SMOTE is incorporated into the training workflow to address class imbalance.

### Behavioural Feature Engineering

`DaysSinceLastPurchase` is derived from customer purchase history to provide additional behavioural information.

### Interactive Analytics

Streamlit transforms the machine learning workflow into an interactive dashboard.

### Model-Based Prediction

A serialized trained model is loaded by the application to generate predictions from new customer inputs.

---

# Learning Outcomes

This project demonstrates practical experience with:

- Exploratory Data Analysis
- Data Cleaning
- Missing-Value Treatment
- Duplicate Removal
- Feature Engineering
- Outlier Detection
- IQR-Based Outlier Handling
- Label Encoding
- One-Hot Encoding
- Feature Scaling
- SMOTE
- Binary Classification
- Model Comparison
- Classification Metrics
- Model Serialization
- Streamlit Development
- Business Analytics

---

# Future Enhancements

Potential improvements for future versions include:

- Add probability calibration
- Add ROC curves and Precision-Recall curves
- Add confusion matrices for individual models
- Add feature-importance analysis
- Add SHAP-based model explainability
- Add customer-level risk categories
- Add batch prediction through CSV upload
- Add automated model selection
- Add cross-validation
- Add hyperparameter optimization
- Add model monitoring
- Deploy the application to a production environment

---

# Limitations

- Predictions depend on the quality and representativeness of the training data.
- Model performance can vary across different datasets.
- Churn predictions represent probabilities or classifications, not guaranteed future behaviour.
- The current prediction interface exposes a subset of the full training feature space and aligns the input with the saved feature columns.
- Further validation would be required before using the system for real-world customer-retention decisions.

---

# Project Status

**Completed**

The project includes data preprocessing, exploratory analysis, multiple classification models, evaluation, model persistence, and an interactive Streamlit dashboard for customer churn analysis and prediction.

---

# Author

**Darshan S**

GitHub:

```text
https://github.com/DarshanS2004
```

---

# License

This project is intended for educational, learning, and portfolio purposes.