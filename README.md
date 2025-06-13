# Heart Disease Prediction

Welcome to my first Machine Learning project! This objective of this project is to select an optimal classification machine learning algorithm to predict the presence of heart disease in patients (based on various clinical features). The workflow includes data exploration, feature engineering, model training, and visual evaluation.

## Dataset

The dataset is a CSV file containing medical and demographic information. The target variable is `HeartDisease`, a binary indicator (1 = presence of heart disease, 0 = absence).

**Example features include:**

Attribute Information
- Age: age of the patient [years]
- Sex: sex of the patient [M: Male, F: Female]
- ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
- RestingBP: resting blood pressure [mm Hg]
- Cholesterol: serum cholesterol [mm/dl]
- FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
- RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left  ventricular hypertrophy by Estes' criteria]
- MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
- ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
- Oldpeak: oldpeak = ST [Numeric value measured in depression]
- ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
- HeartDisease: output class [1: heart disease, 0: Normal]

This dataset was created by combining different datasets already available independently but not combined before. In this dataset, 5 heart datasets are combined over 11 common features which makes it the largest heart disease dataset available so far for research purposes. The five datasets used for its curation are:

Cleveland: 303 observations
Hungarian: 294 observations
Switzerland: 123 observations
Long Beach VA: 200 observations
Stalog (Heart) Data Set: 270 observations
Total: 1190 observations

Every dataset used can be found under the Index of heart disease datasets from UCI Machine Learning Repository on the following link: 

https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/

Link to Kaggle dataset:

https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

---

## Project Workflow

### 1. Data Loading and Initial Exploration

- Load the dataset using pandas
- Preview the structure using `.head()`, `.info()`, and `.describe()`
- Check for and report missing values

### 2. Data Cleaning

- Remove duplicate rows
- Identify numeric and categorical features for visualization and preprocessing

### 3. Exploratory Data Analysis (EDA)

- **Numeric features:** Visualized using histograms
- **Categorical features:** Visualized using count plots
- Designed to give a quick understanding of feature distributions

### 4. Feature Engineering

- Create new features, e.g., `age_bin` for age groups
- Encode categorical variables using one-hot encoding
- Compute correlation with the `HeartDisease` target
- Select features with a strong correlation to improve model performance

---

## Model Training and Evaluation

This project uses four supervised machine learning algorithms to classify patients as having or not having heart disease. Each model is trained, evaluated, and compared using consistent methods and metrics.

### Models Used

1. **Logistic Regression**  
   A linear model used as a strong baseline for binary classification problems.

2. **Random Forest Classifier**  
   An ensemble of decision trees that reduces overfitting and improves predictive power.

3. **Support Vector Machine (SVM)**  
   Finds the optimal hyperplane to separate classes, effective for high-dimensional spaces.

4. **K-Nearest Neighbors (KNN)**  
   A distance-based model that classifies a point based on the majority label of its neighbors.

---

### Training Workflow

1. **Train-Test Split**
   - The dataset is split into 80% training and 20% testing sets using `train_test_split`.
   - Stratification ensures class proportions are consistent between training and testing.

2. **Feature Scaling**
   - StandardScaler is used to normalize the feature values.
   - Scaling is critical for distance-based models (like SVM and KNN) and improves model convergence.

3. **Model Fitting**
   - Each model is trained on the scaled training data.
   - The models use default parameters unless otherwise noted.

4. **Prediction**
   - Predictions are made on the test data.
   - For models that support it, prediction probabilities are used to evaluate ROC-AUC.

---

### Evaluation Metrics

Each model is evaluated on the test set using the following metrics:

- **Accuracy**: Proportion of total correct predictions
- **Precision**: True positives / predicted positives (measures false positives)
- **Recall**: True positives / actual positives (measures false negatives)
- **F1 Score**: Harmonic mean of precision and recall (balances both)
- **AUC (Area Under the Curve)**: Measures classifier performance across all thresholds using the ROC curve

---

### Visualization

The project produces the following visual outputs for each model:

- **Confusion Matrix**  
  A heatmap showing true vs. predicted labels to analyze misclassifications.

- **ROC Curve**  
  Plots True Positive Rate (Recall) vs. False Positive Rate to visualize the trade-off at various thresholds.
  - Each model's AUC score is displayed to summarize performance.

All confusion matrices and ROC curves are presented in **tiled grid layouts** for clear side-by-side comparison.

---

### Summary Output

At the end of the script, a summary table shows each model’s:
- Accuracy
- Precision
- Recall
- F1 Score
- AUC

This allows for quick comparison and selection of the best-performing classifier based on the problem requirements.


### Results Analysis and Model Recommendation

**See the Model Comparison Table Below:**

| Model                | Accuracy | Precision | Recall  | F1     | AUC    |
|----------------------|----------|-----------|---------|--------|--------|
| Logistic Regression  | 0.8859   | 0.8716    | 0.9314  | 0.9005 | 0.9297 |
| Random Forest        | 0.8641   | 0.8667    | 0.8922  | 0.8792 | 0.9323 |
| SVM                  | 0.9022   | 0.8818    | 0.9510  | 0.9151 | 0.9443 |
| KNN                  | 0.8859   | 0.8857    | 0.9118  | 0.8986 | 0.9360 |

The model comparison summary indicates that all four models—Logistic Regression, Random Forest, SVM, and KNN—performed well in predicting heart disease, each achieving high accuracy and F1 scores. Among them, **SVM (Support Vector Machine)** demonstrated the highest overall performance, with an accuracy of 0.9022, recall of 0.9510, F1 score of 0.9151, and the top AUC of 0.9443. This suggests SVM is especially strong at correctly identifying true positive cases, which is critical in medical applications where missing positive cases could have serious consequences.

KNN and Logistic Regression also achieved strong results, with accuracies and F1 scores above 0.88. Logistic Regression, with its strong interpretability and a recall of 0.9314, is a suitable alternative where model transparency is needed. Random Forest performed well, but slightly lagged behind the others in terms of recall and F1, despite its robustness to overfitting.

**Recommendation:**  
Given the importance of sensitivity (recall) and overall discriminative power (AUC) in heart disease prediction, the **SVM model is recommended as the optimal choice**. For applications prioritizing interpretability, Logistic Regression is a strong secondary option. Further hyperparameter tuning and validation on larger datasets are suggested to confirm these findings before deployment.



## How to Run

1. Make sure the following packages are installed:

```bash
pip install pandas matplotlib seaborn scikit-learn
```
2. Download heart.csv from respository
3. Replace file path to load downloaded dataset (Line 15)
4. Run code 

