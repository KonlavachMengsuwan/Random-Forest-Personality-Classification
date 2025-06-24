# Personality Prediction using Random Forest

## 📌 Introduction

This project explores the prediction of personality traits (Introvert vs. Extrovert) using a Random Forest classifier on a synthetic dataset. The aim is to build, evaluate, and interpret a machine learning model that can distinguish personality types based on behavioral and social traits.

---

## 🔬 Research Gap

While personality prediction is a well-studied topic in psychology, it remains underexplored with small-scale behavioral data in a structured tabular format. Most work is done with text data or psychological assessments. This project explores how machine learning can work on numeric and categorical behavioral indicators.

---

## 🎯 Objectives

* Classify personality as Introvert or Extrovert
* Apply Random Forest classifier
* Evaluate baseline and optimized models
* Tune hyperparameters for best accuracy
* Use visualizations to interpret model performance

---

## ⚙️ Methods

* Data cleaning and encoding
* Exploratory Data Analysis (EDA)
* Train/Test split
* Random Forest Classification
* GridSearchCV for hyperparameter tuning
* Performance metrics: confusion matrix, classification report
* Feature importance visualization

---

## 💻 Code Summary

### 1. Install Required Packages

```python
!pip install pandas numpy matplotlib seaborn scikit-learn shap -q
```

### 2. Import Libraries

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
```

### 3. Load & Preprocess Dataset

```python
from google.colab import files
uploaded = files.upload()
df = pd.read_csv("personality_datasert.csv")
df.columns = df.columns.str.strip().str.replace(" ", "_")
```

![image](https://github.com/user-attachments/assets/72516bca-e8be-4d76-8120-dd572bb55634)
![image](https://github.com/user-attachments/assets/acb39ed8-b604-419d-b7d0-5e04d27864b3)
![image](https://github.com/user-attachments/assets/e310146b-8465-4ab6-b48c-3e4b16ed807e)


### 4. Encode Target & Binary Columns

```python
binary_cols = ['Stage_fear', 'Drained_after_socializing']
multi_cols = ['Personality']
for col in binary_cols + multi_cols:
    df[col] = LabelEncoder().fit_transform(df[col])
```

### 5. Train-Test Split

```python
X = df.drop(columns=['Personality'])
y = df['Personality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
```

### 6. Train Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### 7. Evaluate Performance

```python
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

### 8. Feature Importance

```python
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
sns.barplot(data=importance_df, x='Importance', y='Feature')
plt.title("Feature Importance")
plt.show()
```
![image](https://github.com/user-attachments/assets/1d0428e0-ecf0-421f-82c6-a2db86e07a0e)



### 9. Hyperparameter Tuning (Grid Search)

```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

### 10. Visualization of Results

```python
results_df = pd.DataFrame(grid_search.cv_results_)
sns.lineplot(x='param_n_estimators', y='mean_test_score', data=results_df)
plt.title("Effect of Number of Trees")
plt.show()
```

![image](https://github.com/user-attachments/assets/fa397fd5-4988-4ad8-9b5a-893d5d8a5c83)
![image](https://github.com/user-attachments/assets/d96c679e-3fd3-45c2-86d4-41b4bc567a79)
![image](https://github.com/user-attachments/assets/c3bfff39-f1e5-4e90-94ec-eadb513d18d3)
![image](https://github.com/user-attachments/assets/991d7aa0-0150-454b-a6c7-11771bc3d78d)



---

## 📈 Results

### Confusion Matrix

|        | Predicted 0 | Predicted 1 |
| ------ | ----------- | ----------- |
| True 0 | 76          | 4           |
| True 1 | 5           | 75          |

### Classification Report

| Metric    | Class 0 | Class 1 | Accuracy | Macro Avg | Weighted Avg |
| --------- | ------- | ------- | -------- | --------- | ------------ |
| Precision | 0.94    | 0.95    | 0.95     | 0.95      | 0.95         |
| Recall    | 0.95    | 0.94    |          | 0.95      | 0.95         |
| F1-score  | 0.95    | 0.95    |          | 0.95      | 0.95         |
| Support   | 80      | 80      | 160      | 160       | 160          |

---

## 📊 Hyperparameter Insights

* `n_estimators`: More trees generally improve performance but increase training time.
* `max_depth`: Limits tree growth. Prevents overfitting.
* `min_samples_split`: Minimum number of samples to split a node.
* `min_samples_leaf`: Minimum samples at a leaf node.

Heatmaps and line plots revealed optimal values that boosted accuracy from 91% to 95%.

---

![image](https://github.com/user-attachments/assets/fad73b5e-f93d-4047-823a-10810f8c3927)
![image](https://github.com/user-attachments/assets/af95543c-b3a9-477d-8f1b-3686672279bf)

## 🧠 Next Steps

* Add SHAP explainability for per-feature impact
* Compare other models (e.g., SVM, XGBoost)
* Consider collecting real-world behavioral data

---

Code: https://colab.research.google.com/drive/1K96GZDWekVVYaMcq5iJvKjz8nB4f5f2f?usp=sharing


## 📎 License

This project is released under the MIT License.

---

*Developed and maintained by \[Konlavach Mengsuwan]*
