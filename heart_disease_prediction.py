# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 21:51:04 2025

@author: cliff
"""

r"""C:\Users\cliff\OneDrive\Documents\Machine Learning\heart.csv"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r"C:\Users\cliff\OneDrive\Documents\Machine Learning\heart.csv")



# ----- Initial Dataset Description -----
print("Dataset Description\n")
print("First 5 rows:")
print(df.head(), "\n")

print("Data Types and Non-Null Counts:")
print(df.info(), "\n")

print("Summary Statistics:")
print(df.describe(), "\n")

print("Missing Values per Column:")
print(df.isnull().sum(), "\n")



# ----- Data Cleaning -----
df.drop_duplicates(inplace=True)

# ----- Feature Type Detection -----
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns




# ----- Univariate Visualizations -----

# Histograms for numeric features
num_features = len(numeric_cols)
ncols = 3
nrows = (num_features + ncols - 1) // ncols

plt.figure(figsize=(ncols * 5, nrows * 4))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(nrows, ncols, i)
    plt.hist(df[col], bins=20, edgecolor='black')
    plt.title(col)
plt.suptitle('Histograms of Numeric Features', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Count plots for categorical features
cat_features = len(categorical_cols)
if cat_features > 0:
    ncols = 3
    nrows = (cat_features + ncols - 1) // ncols
    plt.figure(figsize=(ncols * 5, nrows * 4))
    for i, col in enumerate(categorical_cols, 1):
        plt.subplot(nrows, ncols, i)
        sns.countplot(x=col, data=df)
        plt.title(col)
    plt.suptitle('Count Plots of Categorical Features', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
else:
    print("No categorical features found.\n")



# ----- Feature Engineering -----
print()
print()
print("Feature Engineering:")
print()

# 1. Create new features
if 'age' in df.columns:
    df['age_bin'] = pd.cut(df['age'], bins=[0, 40, 55, 70, 120], labels=['Young', 'Mid-Age', 'Senior', 'Old'])

# 2. Encode categorical columns (including new 'age_bin' if added)
df_encoded = pd.get_dummies(df, drop_first=True)

# 3. Select important features via correlation with target
if 'target' in df_encoded.columns:
    corr_with_target = df_encoded.corr()['target'].abs().sort_values(ascending=False)
    print("\nTop features correlated with target:")
    print(corr_with_target[1:])  # Exclude correlation of target with itself

    selected_features = corr_with_target[corr_with_target > 0.1].index.tolist()
    selected_features.remove('target')
    df_selected = df_encoded[selected_features + ['target']]
    print(f"\nSelected {len(selected_features)} important features.")
else:
    print("No 'target' column found for feature selection.")
    df_selected = df_encoded.copy()

# Final dataset info
print("\nShape of dataset after feature engineering:", df_selected.shape)
print("Final columns:")
print(df_selected.columns.tolist())

print()
print()



# ----- MODEL TRAINING & EVALUATION -----

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# ----- Define Feature Matrix and Target -----
X = df_selected.drop('HeartDisease', axis=1)
y = df_selected['HeartDisease']

# ----- Train-Test Split -----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ----- Feature Scaling -----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----- Models to Compare -----
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

# ----- Evaluation Function (returns metrics & plots) -----
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr = None, None
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)

    return {
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC": auc,
        "ConfusionMatrix": cm,
        "FPR": fpr,
        "TPR": tpr
    }

# ----- Train and Evaluate All Models -----
results = []
for name, model in models.items():
    res = evaluate_model(name, model, X_train_scaled, y_train, X_test_scaled, y_test)
    results.append(res)

# ----- Summary Table -----
results_df = pd.DataFrame(results)[["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"]].set_index("Model")
print("Model Comparison Summary:")
print(results_df.sort_values(by='F1', ascending=False))

# ----- Plot All Confusion Matrices -----
n = len(results)
cols = 2
rows = (n + 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
axes = axes.flatten()

for i, res in enumerate(results):
    sns.heatmap(res["ConfusionMatrix"], annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'{res["Model"]} Confusion Matrix')
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("Actual")

# Hide unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.suptitle('Confusion Matrices for All Models', fontsize=16)
plt.show()

# ----- Plot All ROC Curves -----
fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
axes = axes.flatten()

for i, res in enumerate(results):
    if res["FPR"] is not None and res["TPR"] is not None:
        axes[i].plot(res["FPR"], res["TPR"], label=f'AUC = {res["AUC"]:.2f}')
        axes[i].plot([0, 1], [0, 1], linestyle='--')
        axes[i].set_title(f'{res["Model"]} ROC Curve')
        axes[i].set_xlabel("False Positive Rate")
        axes[i].set_ylabel("True Positive Rate")
        axes[i].legend()
        axes[i].grid(True)

# Hide unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.suptitle('ROC Curves for All Models', fontsize=16)
plt.show()





    
    
    
    
    




