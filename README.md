# 📊 Customer Churn Prediction Using Machine Learning

This project predicts whether a customer is likely to churn (leave the service) based on historical behavior and service usage.  
The model is built using Python, Pandas, NumPy, Scikit-Learn, Seaborn/Matplotlib, and TensorFlow/Keras.

---

## 🗂️ Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Tech Stack](#tech-stack)  
- [Data Processing Steps](#data-processing-steps-pipeline)  
- [Visualization](#visualization)  
- [Machine Learning Model](#machine-learning-model)  
- [Model Evaluation](#model-evaluation)  
  - [Confusion Matrix](#confusion-matrix)  
  - [Model Performance Chart](#model-performance-chart)  
- [Final Outcome](#final-outcome)  

---
<a name="project-overview"></a>
## 🚀 Project Overview

Customer churn prediction helps companies identify customers who are likely to stop using the service.  
By predicting churn early, companies can take action to retain customers.

This project includes:

- Data cleaning  
- Feature engineering  
- Data visualization  
- Machine learning and deep learning model training  
- Evaluation using confusion matrix and metrics  
- Final prediction




---
<a name="dataset"></a>
 
## 📁 Dataset


The dataset used is the **Telco Customer Churn** dataset, which contains customer demographics, account details, and service usage patterns.

Key columns:

- Gender  
- SeniorCitizen  
- Partner  
- Dependents  
- Tenure  
- Phone & Internet services  
- Contract type  
- Payment method  
- Monthly & total charges  
- Churn (target variable)





---
<a name="tech-stack"></a>

## 🛠️ Tech Stack


- Python  
- Pandas    
- Matplotlib  
- Seaborn  
- Scikit-Learn  
- TensorFlow / Keras  
- Jupyter Notebook / VS Code  




---
<a name="data-processing-steps-pipeline"></a>


## 🔧 Data Processing Steps (Pipeline)

1. Load dataset  
2. Handle missing or incorrect values  
3. Convert `"TotalCharges"` to numeric  
4. Remove irrelevant columns (`customerID`)  
5. Replace service-related categorical values  
6. Encode categorical columns  
7. Scale numerical features  
8. Train-test split  
9. Model training  
10. Model evaluation  




---
<a name="visualization"></a>


## 📉 Visualization


### **Customer Churn Distribution**
![Customer Churn Visualization](https://raw.githubusercontent.com/Atharv-M-Patil/Customer-Churn-Predication/main/customer_churn_prediction.png)  



---

<a name="machine-learning-model"></a>


## 🧠 Machine Learning Model

The final model was built using **TensorFlow/Keras (Artificial Neural Network)**.  
- Dataset was split into train and test sets.  
- The ANN was trained to predict whether a customer will churn.  

**Model Output:**  
- 1 → Customer will churn  
- 0 → Customer will not churn  




---
<a name="model-evaluation"></a>

## 🧪 Model Evaluation

<a name="confusion-matrix"></a>

### **Confusion Matrix**
![Confusion Matrix](https://raw.githubusercontent.com/Atharv-M-Patil/Customer-Churn-Predication/main/confusion_matrix.png)  



<a name="model-performance-chart"></a>

### **Model Performance Chart**
![Model Performance](https://raw.githubusercontent.com/Atharv-M-Patil/Customer-Churn-Predication/main/model_performance.png)



### **Classification Metrics**

| Metric     | Class 0 (No Churn) | Class 1 (Churn) |
|-----------|-------------------|----------------|
| Precision | 0.83              | 0.66           |
| Recall    | 0.88              | 0.54           |
| F1-Score  | 0.85              | 0.59           |
| Support   | 999               | 408            |

**Overall Accuracy:** **0.78 (78%)**  
**Macro Avg F1-Score:** 0.72  
**Weighted Avg F1-Score:** 0.78  



---

<a name="final-outcome"></a>


## 📈 Final Outcome

- The model predicts **non-churn customers (Class 0)** very accurately.  
- Moderate performance for **churn customers (Class 1)** due to class imbalance.  
- Accuracy of **78%** demonstrates reliable churn predictions.  

**Business Impact:**  
- Reduce customer loss  
- Improve retention strategies  
- Understand key churn factors  

---


