# Telco Customer Churn Prediction

## 📌 Project Overview
This project aims to predict customer churn for a telecommunications company using machine learning. By analyzing customer demographics, account information, and service usage, we build models to identify customers likely to leave.

The analysis compares two major classification algorithms: **Logistic Regression** and **Random Forest**.

## 📂 Dataset
The dataset used is the Telco Customer Churn dataset.
* **Source:** [Telco Customer Churn on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) 
* **Target Variable:** `Churn` (Yes/No).
* **Key Features:** Tenure, Monthly Charges, Total Charges, Contract Type, Payment Method, Internet Service, etc.

## 🛠️ Technologies Used
* **Python**
* **Pandas** (Data Manipulation)
* **NumPy** (Numerical Operations)
* **Matplotlib & Seaborn** (Data Visualization)
* **Scikit-Learn** (Machine Learning & Metrics)

## 📊 Methodology
1.  **Data Preprocessing:**
    * Handled missing values in `TotalCharges` by filling with the median.
    * Removed customer IDs as they are non-predictive.
    * Performed **One-Hot Encoding** on categorical variables (e.g., Gender, Partner, InternetService).
    * Split the data into training (80%) and testing (20%) sets.

2.  **Model Building:**
    * **Logistic Regression:** Implemented using a Pipeline with `StandardScaler` to normalize features.
    * **Random Forest:** Implemented an imbalanced model and a balanced class weight model to handle the uneven churn distribution.

## 📈 Results
We evaluated the models based on Accuracy, F1-Score, and ROC-AUC.

| Model | Accuracy | F1-Score (Churn) | ROC-AUC |
|-----------------------|----------|------------------|---------|
| **Logistic Regression** | **80.7%** | **0.61** | **0.842** |
| Random Forest | 79.0% | 0.56 | 0.826 |

* **Key Insight:** The dataset is imbalanced (Non-churn: 1035 vs. Churn: 374 in test set).
* **Conclusion:** Logistic Regression outperformed Random Forest in this specific case, providing a better balance of precision and recall for detecting churners.

## 🚀 How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/customer-churn-prediction.git](https://github.com/yourusername/customer-churn-prediction.git)
    ```
2.  Install dependencies:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn joblib
    ```
3.  Open the notebook:
    ```bash
    jupyter notebook test_01_churn_model.ipynb
    ```

## 🔮 Future Work
* Hyperparameter tuning (GridSearchCV) for the Random Forest model.
* Feature Engineering (creating new features from existing ones).
* Testing other algorithms like XGBoost or LightGBM.
