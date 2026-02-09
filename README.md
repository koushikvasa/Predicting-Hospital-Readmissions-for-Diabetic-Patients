# Predicting Hospital Readmissions for Diabetic Patients

A data mining project developed at **George Mason University** aimed at identifying 30-day readmission risks among diabetic patients. By utilizing machine learning, this study provides a framework for healthcare professionals to proactively intervene and customize treatment plans.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Key Findings](#key-findings)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

Diabetes affects approximately **10% of Americans** and significantly increases the risk of hospital readmission. Hospital readmissions are a significant concern in healthcare, often indicating inadequate initial treatment or insufficient post-discharge care. 

This project focuses on determining whether a patient will be readmitted:
- **Within 30 days** (`<30`)
- **After 30 days** (`>30`)
- **Not at all** (`NO`)

By analyzing clinical characteristics and historical healthcare utilization, this project aims to:
- Predict the likelihood of diabetic patients being readmitted to the hospital
- Identify key factors that contribute to readmissions
- Help healthcare providers implement preventive measures for high-risk patients

## 📊 Dataset

The project uses the **Diabetes 130-US Hospitals for Years 1999-2008** dataset from the **UC Irvine Machine Learning Repository**:

- **Scope**: Data from 130 US hospitals over a 10-year period (1999-2008)
- **Size**: 101,766 patients with 47 initial features
- **Key Features**:
  - **Demographics**: Race, gender, age
  - **Admission details**: Admission type, discharge disposition, admission source
  - **Clinical metrics**: Time in hospital, number of lab procedures, number of procedures
  - **Medical history**: Primary, secondary, and additional diagnoses (`diag_1`, `diag_2`, `diag_3`)
  - **Medications**: Specific diabetic medications including Insulin, Metformin, and others
  - **Healthcare utilization**: Number of outpatient visits, emergency visits, inpatient visits

### Target Variable

The `readmitted` variable indicates whether a patient was readmitted:
- `<30`: Readmitted within 30 days
- `>30`: Readmitted after 30 days
- `NO`: Not readmitted

## 📁 Project Structure

```
Predicting-Hospital-Readmissions-for-Diabetic-Patients/
│
├── Predicting Hospital Readmissions Code.ipynb    # Main Jupyter notebook
├── Predicting Hospital Readmissions for Diabetic Patients.pdf    # Project report
├── README.md                                       # This file
└── data/
    └── diabetic_data.csv                          # Dataset (not included)
```

## 🔧 Requirements

- Python 3.7+
- Jupyter Notebook or JupyterLab

### Python Libraries

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## 💻 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Predicting-Hospital-Readmissions-for-Diabetic-Patients.git
   cd Predicting-Hospital-Readmissions-for-Diabetic-Patients
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```

4. **Download the dataset**:
   - Download the Diabetes 130-US Hospitals dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)
   - Place `diabetic_data.csv` in the project directory
   - Update the file path in the notebook if necessary

## 🚀 Usage

1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the notebook**:
   - Navigate to `Predicting Hospital Readmissions Code.ipynb`

3. **Update the file path** in the second cell to point to your dataset location:
   ```python
   file_path = 'path/to/your/diabetic_data.csv'
   ```

4. **Run all cells** to:
   - Load and explore the data
   - Perform data preprocessing
   - Train machine learning models
   - Evaluate model performance
   - Visualize results

## 🔬 Methodology

### 1. Data Preprocessing

#### Missing Values
- Converted '?' placeholders to NaN values
- Handled missing data through standardization techniques

#### Feature Selection
- **Removed irrelevant identifiers**: Encounter ID, Patient Number
- **Removed high null value columns**: Weight, Max Glu Serum, A1C Result (due to excessive missing data that would compromise model integrity)
- **Removed low variance columns**: Columns with mostly the same value across records were dropped as they contribute minimal information:
  - Medical_specialty, Patient_nbr
  - Medication columns: acetohexamide, tolbutamide, examide, citoglipton
  - Combination medications: glyburide-metformin, glipizide-metformin, glimepiride-pioglitazone, metformin-rosiglitazone, metformin-pioglitazone

#### Imputation
- **Continuous Variables** (`diag_1`, `diag_2`, `diag_3`):
  - Applied **KNN Imputer** with n_neighbors=5
  - Replaced missing values based on similar data points
  
- **Categorical Variables** (`race`):
  - Used **Mode Imputation** (most frequent value)

#### Encoding
- **Ordinal Encoder**: Converted categorical data into unique integers for model interpretation
- Ensured all categorical variables were properly transformed for machine learning algorithms

#### Standardization
- Implemented **StandardScaler** to normalize features
- Achieved mean of zero and standard deviation of one across all features
- Essential for distance-based algorithms and ensuring fair feature contribution

### 2. Dimensionality Reduction

**Principal Component Analysis (PCA)** was applied to:
- Lower data dimensionality
- Reduce noise in the dataset
- Mitigate overfitting
- Improve model generalization
- Enhance computational efficiency

### 3. Model Training & Evaluation

- **Train-Test Split**: 80/20 ratio (80% training, 20% testing)
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
- **Comparison**: Models evaluated both with and without PCA to assess dimensionality reduction impact

## 🤖 Models Implemented

The following machine learning algorithms were trained and evaluated:

### 1. **Naive Bayes (Gaussian)**
- Probabilistic classifier based on Bayes' theorem
- Assumes feature independence
- Fast training and prediction
- Good baseline for continuous features

### 2. **Random Forest Classifier**
- Ensemble learning method using multiple decision trees
- Reduces overfitting through bagging
- Handles non-linear relationships effectively
- Provides feature importance scores

### 3. **Decision Tree**
- Tree-based learning algorithm
- Interpretable model structure
- Handles both categorical and numerical data
- Prone to overfitting without pruning

### 4. **Gradient Boosting**
- Sequential ensemble technique
- Builds trees one at a time, correcting previous errors
- Generally achieves high accuracy
- More computationally intensive than Random Forest

## 📈 Results

### Model Performance Comparison

The use of **PCA generally enhanced performance** across most models. Below is a comprehensive comparison of all evaluation metrics with and without PCA:

#### Without PCA

| Model              | Precision | Recall | F1-Score | Accuracy |
|--------------------|-----------|--------|----------|----------|
| Naive Bayes        | 0.49      | 0.54   | 0.44     | 0.54     |
| Random Forest      | 0.55      | 0.58   | 0.53     | 0.58     |
| Decision Tree      | 0.45      | 0.44   | 0.45     | 0.44     |
| Gradient Boosting  | 0.50      | 0.56   | 0.48     | 0.55     |

#### With PCA

| Model              | Precision | Recall | F1-Score | Accuracy |
|--------------------|-----------|--------|----------|----------|
| Naive Bayes        | 0.59      | 0.57   | 0.49     | 0.57     |
| Random Forest      | 0.57      | 0.57   | 0.57     | 0.57     |
| Decision Tree      | 0.53      | 0.54   | 0.53     | 0.53     |
| **Gradient Boosting** | **0.60** | **0.60** | **0.60** | **0.60** |

#### Performance Improvement Summary

| Model              | Accuracy (No PCA) | Accuracy (With PCA) | Improvement |
|--------------------|-------------------|---------------------|-------------|
| **Gradient Boosting** | 55%            | **60%**             | **+5%**     |
| **Random Forest**     | 58%            | 57%                 | -1%         |
| **Naive Bayes**       | 54%            | 57%                 | **+3%**     |
| **Decision Tree**     | 44%            | 53%                 | **+9%**     |

### Key Performance Insights

🏆 **Best Model**: **Gradient Boosting with PCA** achieved the highest overall performance with:
- **60% Accuracy**
- **0.60 Precision** (fewest false positives)
- **0.60 Recall** (best at identifying actual readmissions)
- **0.60 F1-Score** (perfect balance between precision and recall)

**Notable Observations**:
- **Decision Tree** showed the most significant improvement (**+9%**) with PCA, jumping from 44% to 53% accuracy
- **Gradient Boosting** consistently performed well, achieving the best results with PCA across all metrics
- **Random Forest** performed slightly better without PCA (58% vs 57%), likely due to its inherent ability to handle high-dimensional data
- **Naive Bayes** benefited from dimensionality reduction, improving by 3% and achieving balanced precision-recall
- All models with PCA (except Random Forest) showed improved or maintained performance

### Evaluation Metrics Explained

The project utilized four key metrics to assess model performance:

- **Accuracy**: Overall correctness in predicting whether a patient will be readmitted within 30 days. Critical for demonstrating the model's effectiveness in real-world clinical scenarios.

- **Precision**: Accuracy of positive predictions (of all predicted readmissions, how many were correct). Essential in medical predictions to reduce unnecessary treatments or interventions.

- **Recall (Sensitivity)**: Model's ability to identify all actual positives (of all actual readmissions, how many did we catch). High recall ensures most at-risk patients are identified for appropriate follow-up treatment.

- **F1 Score**: Harmonic mean of precision and recall, providing balance between the two. Especially useful given potential imbalance between readmitted and non-readmitted patient groups.

### Why PCA Improved Performance

PCA contributed to better model performance by:
1. **Reducing noise** in the high-dimensional dataset (47 features reduced to principal components)
2. **Eliminating multicollinearity** between correlated features
3. **Mitigating overfitting** by focusing on components with highest variance
4. **Improving generalization** to unseen data
5. **Enhancing computational efficiency** during training

## 🔍 Key Findings

### Important Predictive Features

The most significant features for predicting hospital readmission include:

1. **Healthcare Utilization**
   - Number of inpatient visits
   - Number of emergency visits
   - Number of outpatient visits

2. **Clinical Metrics**
   - Time spent in hospital
   - Number of lab procedures
   - Number of procedures performed

3. **Diagnoses**
   - Primary diagnosis (`diag_1`)
   - Secondary diagnosis (`diag_2`)
   - Additional diagnosis (`diag_3`)

4. **Medications**
   - Insulin administration
   - Other diabetic medications (Metformin, etc.)
   - Number of medications prescribed

5. **Demographics & Admission Type**
   - Age group
   - Admission type (Emergency, Urgent, Elective)
   - Discharge disposition

### Clinical Insights

- **Patients with multiple previous admissions** are at significantly higher risk of readmission
- **Longer hospital stays** correlate with increased readmission risk
- **Specific diagnosis codes** show strong predictive power for 30-day readmission
- **Age and number of medications** are significant contributing factors
- **Emergency admissions** tend to have higher readmission rates than elective procedures

## 📖 Related Work

The challenge of predicting hospital readmissions for diabetic patients has received widespread attention due to its important implications for patient care and healthcare system efficiency. Our study builds upon several seminal contributions:

### Key Studies

**Strack et al. (2014)** - Pioneering Study
- Created predictive models to predict diabetic patients' chance of hospital readmission within 30 days
- Used logistic regression emphasizing the importance of clinical and demographic data
- Established a standard for accuracy in predicting readmissions, influencing future research in the field

**Shah et al. (2015)** - Machine Learning Approaches
- Investigated various machine learning approaches including Decision Trees, Support Vector Machines, and Neural Networks
- Highlighted the importance of algorithmic complexity and data quality in achieving high prediction accuracy
- Suggested a trade-off between model simplicity and performance

**Kaur and Kumari (2018)** - Feature Selection Impact
- Investigated the effect of reducing feature space on predictive model performance
- Found that accurate feature selection can considerably increase the performance of models such as Random Forests and Naive Bayes
- Aligned with our findings on dimensionality reduction benefits

### Our Contribution

This project expands on past research by:
- Comparing the efficacy of multiple machine learning models (Decision Tree, Random Forest, Naive Bayes, Gradient Boosting)
- Evaluating models both with and without PCA to determine the most efficient method
- Demonstrating that dimensionality reduction significantly improves model performance
- Contributing to the ongoing discussion in healthcare analytics with the goal of improving predicted accuracy and patient care outcomes

## 💡 Conclusion

This project demonstrated the effectiveness of data mining techniques in predicting 30-day readmissions for diabetic patients, which is crucial for improving healthcare management. By analyzing a comprehensive dataset from 130 US hospitals spanning 10 years, various machine learning models were rigorously evaluated.

### Key Takeaways

1. **PCA Impact**: The use of Principal Component Analysis generally enhanced model performance, highlighting its utility in reducing dimensionality and mitigating overfitting

2. **Best Performing Model**: Gradient Boosting with PCA achieved the highest accuracy (60%) and demonstrated balanced performance across all metrics (precision, recall, F1-score)

3. **Practical Application**: The evaluation metrics—accuracy, precision, recall, and F1 score—highlighted the models' reliability and practicality in clinical settings, aiding in the precise identification of patients at risk of readmission

4. **Healthcare Impact**: This predictive capability not only helps in better patient care but also reduces costs associated with frequent readmissions

### Future Implications

This project underscores the potential of integrating advanced data analytics into healthcare practices to enhance predictive capabilities. Future work could include:
- Testing additional ensemble methods and deep learning approaches
- Incorporating real-time patient monitoring data
- Expanding to other chronic conditions beyond diabetes
- Developing clinical decision support systems based on these models
- Conducting prospective validation studies in hospital settings

Ultimately, this work suggests further exploration and continuous methodology refinement for broader application in healthcare predictive analytics.

## 👥 Team Members

This project was developed by students at **George Mason University**:

- **Sravya Sri Kodali** (G01460802) - skodali6@gmu.edu
- **Pruthvinath Reddy Sireddy** (G01458015) - psireddy@gmu.edu  
- **Koushik Vasa** (G01480627) - kvasa@gmu.edu

### Division of Work

- **Data Preprocessing**: Pruthvinath Reddy Sireddy
- **Model Training**: Koushik Vasa
- **PowerPoint Presentation**: Pruthvinath Reddy Sireddy, Koushik Vasa, Sravya Sri Kodali
- **Final Report**: Sravya Sri Kodali

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is available for educational and research purposes. Please check the UCI Machine Learning Repository for dataset license terms.

## 📚 References

### Academic Sources

1. **Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014)**. "Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records." *BioMed Research International*, vol. 2014, Article ID 781670. [https://doi.org/10.1155/2014/781670](https://doi.org/10.1155/2014/781670)

2. **Shah, B. R., Hux, J. E., Laupacis, A., Zinman, B., & van Walraven, C. (2005)**. "Clinical inertia in response to inadequate glycemic control: Do specialists differ from primary care physicians?" *Diabetes Care*, 28(3), 600-606. [https://doi.org/10.2337/diacare.28.3.600](https://doi.org/10.2337/diacare.28.3.600)

3. **Kaur, H., & Kumari, V. (2018)**. "Predictive modelling and analytics for diabetes using a machine learning approach." *Applied Computing and Informatics*. [https://doi.org/10.1016/j.aci.2018.08.003](https://doi.org/10.1016/j.aci.2018.08.003)

### Dataset

- UCI Machine Learning Repository: [Diabetes 130-US Hospitals Dataset](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: This project is for educational purposes and should not be used for actual clinical decision-making without proper validation and regulatory approval.
