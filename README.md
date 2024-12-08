# Loan Approval Prediction Model

This repository contains the implementation of a Loan Approval Prediction Model using various machine learning techniques and advanced feature engineering.

---

## Dataset

- **Source:** The dataset used for this project is publicly available on Kaggle: [Loan Approval Classification Data](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data).  
- **Objective:** To predict whether a loan will be approved based on demographic, financial, and credit-related features.

---

## Tools & Libraries Used

- **Languages:** Python  
- **Tools:** Jupyter Notebook  
- **Libraries:** pandas, numpy, matplotlib, seaborn, optbinning, regex, scikit-learn, xgboost, lightgbm, catboost, shap  

---

## Project Steps

### 1. **Data Loading and Initial Exploration**
- Loaded the dataset into a Pandas DataFrame.  
- Performed an initial exploration to understand the dataset structure and missing values.  

### 2. **Exploratory Data Analysis (EDA)**
- **Visualizations:** Created bar charts, histograms, box plots, distplots, and crosstabs.  
- **Feature Analysis:** Examined the distribution of various features with respect to the target variable (`loan_status`).  

---

### 3. **Feature Engineering**
Created new features to capture deeper insights:

#### **Demographic Features**
| **Feature Name**                | **Formula/Method**                                                                                     | **Purpose**                                           |
|----------------------------------|--------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| `Age Brackets`                  | Categorized `person_age` into groups (e.g., `<25`, `25-40`, `40-60`, `>60`).                          | Captures age-related risk patterns.                  |
| `Employment Stability Ratio`    | `person_emp_exp / person_age`.                                                                         | Measures employment stability.                       |
| `Dependence on Income Source`   | `person_income / loan_amnt`.                                                                          | Indicates financial stability.                       |

#### **Credit and Loan Features**
| **Feature Name**                | **Formula/Method**                                                                                     | **Purpose**                                           |
|----------------------------------|--------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| `Credit Utilization Ratio`      | `loan_amnt / (person_income * cb_person_cred_hist_length)`.                                            | Measures credit reliance.                            |
| `Debt-to-Income Ratio`          | `loan_amnt / person_income`.                                                                          | Indicates income commitment to loans.                |
| `Credit Score Bucket`           | Buckets created for `credit_score` (e.g., `Low`, `Medium`, `High`).                                   | Simplifies credit score impact.                      |

#### **Behavioral Features**
| **Feature Name**                | **Formula/Method**                                                                                     | **Purpose**                                           |
|----------------------------------|--------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| `Previous Default Risk`         | Encoded `previous_loan_defaults_on_file` as binary (0: No defaults, 1: Defaults).                     | Tracks default risk.                                 |
| `Loan Intent Risk Factor`       | Mapped `loan_intent` to risk levels (e.g., Education = Low Risk, Small Business = High Risk).          | Categorizes loan purpose risk.                       |

#### **Combined Features**
| **Feature Name**                | **Formula/Method**                                                                                     | **Purpose**                                           |
|----------------------------------|--------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| `Income Stability`              | `person_income * person_emp_exp`.                                                                     | Indicates financial stability.                       |
| `Debt Repayment Capacity`       | `person_income - (loan_amnt * loan_int_rate)`.                                                        | Measures income left after loan payments.            |

#### **Interaction Features**
| **Feature Name**                | **Formula/Method**                                                                                     | **Purpose**                                           |
|----------------------------------|--------------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| `Loan Percent Income Interaction` | `loan_percent_income * cb_person_cred_hist_length`.                                                  | Combines income dependency with credit history.       |


---

## Key Steps in Calculating WOE and IV

### 1. **WOE Calculation for Categorical Variables**

The process includes:
- Generating a frequency table for each categorical variable against the target variable (`loan_status`), categorizing values into "Event" (default) and "Non-Event" (no default).
- Calculating the percentage of "Event" and "Non-Event" for each category of the variable.
- Computing the WOE values using the natural logarithm of the ratio of "Non-Event%" to "Event%."
- Determining the IV values by multiplying the difference between "Non-Event%" and "Event%" with the WOE value.
- Organizing the results into a consolidated DataFrame for all categorical variables.

---

### 2. **WOE Calculation for Numerical Variables**

The process involves:
- Using an optimal binning technique to group numerical variables into bins based on their relationship with the target variable.
- For each bin:
  - Determining the boundaries (lower and upper limits) of the numerical ranges.
  - Calculating WOE and IV values similarly to categorical variables.
- Storing the results in a final table for all numerical features.

---

### 3. **Integration of WOE Values in the Dataset**

- Numerical features are replaced with their corresponding WOE values by checking which bin each value falls into and applying the precomputed WOE value for that bin.
- Categorical features are directly mapped to their WOE values based on the precomputed table.

---

### 4. **Significance of WOE and IV**

- **WOE**:
  - Measures the predictive power of individual bins for a variable.
  - Values closer to zero indicate weaker predictive ability.
- **IV**:
  - Summarizes the predictive strength of the entire variable.
  - Interpretation of IV values:
    - IV < 0.02: Not predictive.
    - 0.02 ≤ IV < 0.1: Weak predictive power.
    - 0.1 ≤ IV < 0.3: Medium predictive power.
    - IV ≥ 0.3: Strong predictive power.

---

This process ensures that the model uses features with high predictive power while avoiding bias introduced by raw numerical or categorical values.



### 4. **Feature Selection**
- **Methods Used:** Correlation matrix, Information Value (IV), decision tree importances, Recursive Feature Elimination (RFE).  

---

### 5. **Model Training**
- **Data Split:** Splitted the data into training and testing sets.  
- **Models Used:**  
  - Logistic Regression  
  - K-Nearest Neighbors  
  - Naive Bayes  
  - Decision Tree  
  - Random Forest  
  - Support Vector Machine  
  - Gradient Boosting  
  - AdaBoost  
  - XGBoost  
  - LightGBM  
  - CatBoost  

| **Model**                 | **Test Accuracy** | **Test Precision** | **Test Recall** | **Test F1-Score** | **Test AUC Score** |
|---------------------------|-------------------|---------------------|-----------------|-------------------|--------------------|
| Logistic Regression       | 0.868922          | 0.862634            | 0.868922        | 0.862069          | 0.882071           |
| K-Nearest Neighbors       | 0.881358          | 0.876555            | 0.881358        | 0.876529          | 0.869548           |
| Naive Bayes               | 0.847188          | 0.847629            | 0.847188        | 0.847405          | 0.863150           |
| Decision Tree             | 0.857047          | 0.855954            | 0.857047        | 0.856475          | 0.799999           |
| Random Forest             | 0.892113          | 0.888325            | 0.892113        | 0.888208          | 0.901099           |
| Gradient Boosting         | 0.892785          | 0.889992            | 0.892785        | 0.886576          | 0.911757           |
| XGBoost                   | 0.898835          | 0.896472            | 0.898835        | 0.893428          | 0.912936           |
| LightGBM                  | 0.899507          | 0.897542            | 0.899507        | 0.893820          | 0.915816           |
| CatBoost                  | 0.899619          | 0.897741            | 0.899619        | 0.893872          | 0.918018           |

---

### 6. **Model Evaluation**
- **Performance Metrics:** Accuracy, Precision, Recall, F1-Score, AUC-ROC.  
- **Visualizations:** ROC Curves, Confusion Matrix, Classification Reports.  

---

### 7. **Model Interpretation**
- **SHAP Analysis:** Used SHAP for model interpretation.  
  - Beeswarm Plot  
  - Waterfall Plot  
  - Bar Plot  

---

### 8. **KS Table Analysis**
### **KS Table**

| **Deciles** | **Pred_Proba_Min** | **Pred_Proba_Max** | **Event** | **Non-Event** | **Event Rate (%)** | **Non-Event Rate (%)** | **Cum. Event Rate (%)** | **Cum. Non-Event Rate (%)** | **KS (%)** |
|-------------|--------------------|--------------------|-----------|---------------|--------------------|-------------------------|-------------------------|-----------------------------|------------|
| 10          | 0.950881          | 0.997682          | 878       | 15            | 44.08              | 0.22                   | 44.08                  | 0.22                       | **43.86**  |
| 9           | 0.500886          | 0.950783          | 530       | 362           | 26.61              | 5.22                   | 70.68                  | 5.44                       | **65.25**  |
| 8           | 0.285875          | 0.500827          | 216       | 677           | 10.84              | 9.76                   | 81.53                  | 15.20                      | **66.33**  |
| 7           | 0.200900          | 0.285428          | 142       | 750           | 7.13               | 10.82                  | 88.65                  | 26.02                      | **62.64**  |
| 6           | 0.147784          | 0.200900          | 73        | 820           | 3.66               | 11.83                  | 92.32                  | 37.84                      | **54.48**  |
| 5           | 0.104562          | 0.147768          | 67        | 825           | 3.36               | 11.90                  | 95.68                  | 49.74                      | **45.94**  |
| 4           | 0.073820          | 0.104530          | 44        | 849           | 2.21               | 12.24                  | 97.89                  | 61.98                      | **35.91**  |
| 3           | 0.043872          | 0.073784          | 31        | 861           | 1.56               | 12.42                  | 99.45                  | 74.40                      | **25.05**  |
| 2           | 0.016396          | 0.043860          | 11        | 882           | 0.55               | 12.72                  | 100.00                 | 87.12                      | **12.88**  |
| 1           | 0.001663          | 0.016395          | 0         | 893           | 0.00               | 12.88                  | 100.00                 | 100.00                     | **0.00**   |

---

### **Key Observations**
1. **Maximum KS Value**: The maximum KS value is **66.33%**, observed at decile 8, indicating a good separation between events and non-events.
2. **Event Rate & Non-Event Rate**: The event rate decreases as we move from decile 10 to 1, while the non-event rate increases.

This KS table demonstrates strong model performance with good discriminatory power.

---

### 9. **Deployment**
- **Pickled Files:**  
  - WOE Values  
  - XGBoost Model  

- **Prediction Function:** Wrote a custom function to predict on new data.  

---

## Future Improvements
- Fine-tuning model parameters.  
- Incorporating ensemble methods for improved performance.  
- Exploring deep learning-based approaches.  

---
