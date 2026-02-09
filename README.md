Predicting Hospital Readmissions for Diabetic Patients
This repository contains the code and documentation for a data mining project focused on predicting 30-day hospital readmissions for patients with diabetes. The project was completed as part of coursework at George Mason University.

Project Overview
The primary goal of this study is to identify diabetic patients at high risk of being readmitted to the hospital within 30 days of discharge. By leveraging machine learning and data mining techniques, healthcare providers can proactively intervene and customize treatment plans to improve patient outcomes and reduce healthcare costs.

Contributors

Koushik Vasa - Model Training 


Pruthvinath Reddy Sireddy - Data Preprocessing 


Sravya Sri Kodali - Final Report 

Dataset
The project utilizes a dataset from the UC Irvine Machine Learning Repository, which includes ten years (1999–2008) of clinical data from 130 US hospitals.


Size: 101,766 patients and 47 features.


Content: Patient demographics (race, gender, age), clinical characteristics (diagnosis, lab results), and historical healthcare utilization.

Methodology
1. Data Preprocessing

Missing Values: Converted '?' placeholders to NaN.


Column Removal: Dropped irrelevant identifiers (e.g., Encounter ID) and features with high null counts (e.g., Weight) or low variance.

Imputation:


Continuous variables (diag_1, diag_2, diag_3): Handled using KNN Imputer.


Categorical variables (race): Handled using Mode Imputation.


Encoding: Categorical variables were transformed using Ordinal Encoding.


Standardization: Features were normalized using StandardScaler to achieve a mean of zero and a standard deviation of one.

2. Dimensionality Reduction
The project extensively compared model performance with and without Principal Component Analysis (PCA) to evaluate the impact of dimensionality reduction on predictive accuracy and overfitting.

3. Machine Learning Models
Four primary models were evaluated:


Decision Tree 


Random Forest 


Naive Bayes 


Gradient Boosting 

Performance Results
The following table summarizes the performance metrics (Accuracy, Precision, Recall, and F1-Score) for the models when used with and without PCA.

Model	Accuracy (No PCA)	Accuracy (With PCA)
Naive Bayes	
54% 

57% 

Random Forest	
58% 

57% 

Decision Tree	
44% 

53% 

Gradient Boosting	
55% 

60% 

Key Findings
The Gradient Boosting model with PCA achieved the highest overall accuracy of 60%.


PCA generally enhanced model performance, particularly for the Decision Tree, which saw an accuracy jump from 44% to 53%.

Repository Structure
Predicting Hospital Readmissions Code.ipynb: The Jupyter Notebook containing data cleaning, imputation, normalization, and model training.

Predicting Hospital Readmissions for Diabetic Patients.pdf: The detailed final project report.

Would you like me to generate a summary of the "Related Work" section or help you refine the code for a specific model?
