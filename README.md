# Telco Customer Churn Prediction

## 📌 Project Overview
This project focuses on predicting customer churn for a telecommunications company. "Churn" refers to the phenomenon where customers stop doing business with a company. By analyzing customer demographics, service usage, and account details, we aim to identify customers at risk of leaving and understand the key drivers of churn.

This analysis compares **Logistic Regression** (as a baseline) against **Random Forest** classifiers, specifically testing whether addressing class imbalance improves model performance.

## 📂 Dataset
The dataset used is the **Telco Customer Churn** dataset.
* **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Target Variable:** `Churn` (Yes/No) - Indication of whether the customer left within the last month.
* **Key Features:**
    * **Demographics:** Gender, SeniorCitizen, Partner, Dependents.
    * **Services:** PhoneService, InternetService, OnlineSecurity, StreamingTV, etc.
    * **Account Info:** Tenure, Contract, PaymentMethod, MonthlyCharges, TotalCharges.

## 🛠️ Technologies Used
* **Python 3.x**
* **Pandas:** Data manipulation and cleaning.
* **NumPy:** Numerical operations.
* **Matplotlib & Seaborn:** Data visualization.
* **Scikit-Learn:** Machine learning models, preprocessing, and evaluation metrics.
* **Joblib:** Model persistence.

## 📊 Methodology
The project follows a standard data science workflow:

### 1. Data Cleaning & Preprocessing
* **Handling Missing Values:** Detected missing values in the `TotalCharges` column and imputed them with the median value.
* **Feature Selection:** Removed `customerID` as it provides no predictive value.
* **Encoding:** Converted categorical variables (like `InternetService`, `Contract`, `PaymentMethod`) into numeric format using One-Hot Encoding/Label Encoding.
* **Splitting:** Divided data into Training (80%) and Testing (20%) sets.

### 2. Model Building
We trained three distinct variations to compare performance:
1.  **Logistic Regression:** Used as a baseline model with `StandardScaler` for feature normalization.
2.  **Random Forest (Standard):** A standard ensemble model.
3.  **Random Forest (Balanced):** A specific experiment to address the class imbalance (fewer churners than non-churners) by setting `class_weight='balanced'`. This assigns higher penalties to misclassifying the minority class.

### 3. Evaluation
Models were evaluated using **Accuracy**, **F1-Score**, and **ROC-AUC (Area Under the Curve)**. ROC-AUC was chosen as the primary metric because it handles imbalanced datasets better than simple accuracy.

## 📈 Results
The experimental results on the test set are as follows:

| Model | Class Weights | Accuracy | ROC-AUC |
|-------|---------------|----------|---------|
| **Logistic Regression** | None | **80.7%** | **0.842** |
| Random Forest | Standard | 79.0% | 0.826 |
| Random Forest | Balanced | 78.9% | 0.825 |

### Key Findings
* **Logistic Regression Performed Best:** Surprisingly, the linear baseline model slightly outperformed the Random Forest models in both Accuracy and AUC.
* **Balancing Didn't Improve AUC:** The "Balanced" Random Forest approach (`class_weight='balanced'`) yielded nearly identical results to the standard Random Forest. This suggests that simply re-weighting the classes was not sufficient to boost performance for this specific dataset and feature set.
* **Imbalanced Data:** The dataset is naturally imbalanced (~73% retain vs ~27% churn). Despite this, the Logistic Regression model proved robust.

## 🚀 How to Run the Project
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Sachinisand/Customer-Churn-Prediction.git
    cd Customer-Churn-Prediction
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn joblib
    ```

3.  **Run the Notebook:**
    ```bash
    jupyter notebook test_01_churn_model.ipynb
    ```

## 🔮 Future Improvements
* **Hyperparameter Tuning:** Use `GridSearchCV` or `RandomizedSearchCV` to optimize the Random Forest parameters (e.g., `n_estimators`, `max_depth`) rather than using defaults.
* **Feature Engineering:** Create new features, such as grouping customers by tenure cohorts (e.g., "New", "Loyal") to see if that improves predictive power.
* **Advanced Algorithms:** Test gradient boosting methods like **XGBoost** or **LightGBM**, which often handle tabular data better than standard Random Forests.

---
## 🙌 Credits
* **Dataset:** [Telco Customer Churn Dataset – Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* **Project Created by:** Sachini Hewahattage
* **Profile:** Master’s in Data Analytics | Machine Learning & NLP
