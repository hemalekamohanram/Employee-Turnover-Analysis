Employee Turnover Analysis
Project Purpose

This project analyzes employee turnover (attrition) to identify factors contributing to employees leaving an organization and builds predictive models to forecast which employees are at high risk of leaving. The goal is to provide actionable insights that HR teams can use to reduce turnover and improve employee retention.

Technologies & Tools

Programming Language: Python

Environment: Jupyter Notebook

Libraries:

Data manipulation: pandas, numpy

Data visualization: matplotlib, seaborn

Machine learning: scikit-learn (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier)

Evaluation metrics: roc_auc_score, confusion_matrix, classification_report

Features & Analyses

Exploratory Data Analysis (EDA):

Visualizations of employee demographics, job satisfaction, work hours, and other factors.

Identification of patterns associated with attrition.

Data Preprocessing:

Handling missing values, categorical encoding, and data balancing (resampling).

Predictive Modeling:

Models: Logistic Regression, Random Forest, Gradient Boosting.

Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.

Comparison of model performance to select the best model.

Insights & Recommendations:

Identification of top features influencing turnover.

Actionable strategies for HR to reduce attrition.

Setup & Installation

Clone the repository:

git clone <repository_url>


Install required Python libraries:

pip install pandas numpy matplotlib seaborn scikit-learn


Open the Jupyter Notebook:

jupyter notebook


Run the notebook cells sequentially to reproduce the analysis.

Usage Example

Load the dataset and perform exploratory analysis:

import pandas as pd
df = pd.read_csv("employee_turnover.csv")
df.head()


Train a predictive model and evaluate performance:

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
