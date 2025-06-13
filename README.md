# Heart Disease Prediction

This is my first Machine Learning project. This project uses machine learning to select an optimal algorithm to predict the presence of heart disease in patients (based on various clinical features). The workflow includes data exploration, feature engineering, model training, and visual evaluation.

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


## How to Run

1. Make sure the following packages are installed:

```bash
pip install pandas matplotlib seaborn scikit-learn
```
2. Download heart.csv from respoitory
3. Replace file path to load dataset (Line 15)
4. Run code 

