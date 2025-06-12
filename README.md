# Heart Disease Prediction

This project uses machine learning to predict the presence of heart disease in patients based on various clinical features. The workflow includes data exploration, feature engineering, model training, and visual evaluation.

## Dataset

The dataset is a CSV file containing medical and demographic information. The target variable is `HeartDisease`, a binary indicator (1 = presence of heart disease, 0 = absence).

**Example features include:**

- Age
- Sex
- Resting blood pressure
- Cholesterol
- Maximum heart rate
- Chest pain type
- Fasting blood sugar
- And more

The dataset is loaded from the following path (update as needed):

C:\Users\cliff\OneDrive\Documents\Machine Learning\heart.csv


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

### Models Compared

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

### Process

- Split data into training and test sets (80/20)
- Scale features using `StandardScaler`
- Train each model on the training data
- Evaluate models on the test data

### Metrics Used

- Accuracy
- Precision
- Recall
- F1 Score
- Area Under the Curve (AUC)

---

## Visual Outputs

Two sets of visualizations are produced for each model:

1. **Confusion Matrices:** Displayed in a tiled grid for side-by-side comparison
2. **ROC Curves:** Tiled layout with AUC displayed for each model

These help evaluate performance and trade-offs across all models at a glance.

---

## Results

A summary table is printed showing all metrics per model, sorted by F1 score to highlight the best-performing classifiers.

---

## How to Run

1. Make sure the following packages are installed:

```bash
pip install pandas matplotlib seaborn scikit-learn
