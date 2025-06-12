Heart Disease Prediction Project
This project focuses on predicting the presence of heart disease using machine learning techniques. It includes steps for data preparation, exploratory analysis, feature engineering, model training, evaluation, and visual comparison of classification results.

Dataset Overview
The dataset used in this project contains various medical and demographic features such as:

Age

Sex

Cholesterol levels

Blood pressure

Other health indicators

The target variable is HeartDisease, indicating whether or not a patient is diagnosed with heart disease (binary: 0 or 1).

The dataset is loaded from:

makefile
Copy
C:\Users\cliff\OneDrive\Documents\Machine Learning\heart.csv
Workflow Summary
1. Data Loading and Initial Exploration
Load the dataset using pandas

Display the first few rows, data types, summary statistics, and missing values

2. Data Cleaning
Remove duplicate records

Separate numeric and categorical features

3. Exploratory Data Analysis (EDA)
Numeric features: Visualized using histograms

Categorical features: Visualized using count plots

4. Feature Engineering
Creation of a new categorical feature: age_bin (based on age ranges)

Encoding of categorical variables using one-hot encoding

Feature selection based on correlation with the target variable

A final dataset is prepared using only features with significant correlation to HeartDisease

Model Training and Evaluation
Four classification models are trained and compared:

Logistic Regression

Random Forest Classifier

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Steps Performed:
The data is split into training and testing sets using an 80/20 split

Features are scaled using StandardScaler

Each model is trained on the training data

Predictions are made on the test data

Evaluation Metrics:
Accuracy

Precision

Recall

F1 Score

Area Under the ROC Curve (AUC)

Visualizations
Two separate visual comparisons are generated:

Confusion Matrices: All confusion matrices for the four models are displayed in a tiled grid layout for easy comparison.

ROC Curves: ROC curves for each model are also plotted in a tiled layout, along with the AUC scores, to visually assess the trade-off between true and false positive rates.

Summary Output
The script provides:

A tabular summary of all models and their evaluation metrics

Side-by-side visual comparison of classification performance through confusion matrices and ROC curves

How to Use
Ensure the required Python packages are installed:

bash
Copy
pip install pandas matplotlib seaborn scikit-learn
Update the file path to heart.csv in the script if needed.

Run the script using a Python IDE such as Spyder, Jupyter Notebook, or directly from a terminal.

Notes
The code is written for easy readability and modular structure, making it simple to extend.

Next steps can include cross-validation, hyperparameter tuning, feature importance analysis, or model deployment.
