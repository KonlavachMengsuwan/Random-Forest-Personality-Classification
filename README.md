# 🧠 Personality Classification with Random Forest

This project applies machine learning techniques to predict **personality types** (introvert or extrovert) based on behavioral, social, and emotional traits. The pipeline uses a **Random Forest Classifier** with extensive **hyperparameter tuning** and visual analysis.

---

## 📂 Dataset Overview

* **File:** `personality_datasert.csv`
* **Target Variable:** `Personality` (0 = Introvert, 1 = Extrovert)
* **Features Include:**

  * Time spent alone
  * Friends circle size
  * Post frequency
  * Stage fear
  * Social energy (Drained after socializing)

---

## 🎯 Objectives

* Clean and explore the dataset
* Build a predictive model to classify personality
* Visualize important features and tuning impact
* Identify optimal hyperparameters via GridSearchCV

---

## 🧪 Methods

1. **Data Preprocessing**

   * Clean column names
   * Encode binary/categorical columns using `LabelEncoder`
2. **Exploratory Data Analysis**

   * Distribution plots
   * Correlation heatmap
3. **Model Training**

   * `RandomForestClassifier` with default and tuned hyperparameters
4. **Evaluation**

   * Confusion matrix
   * Classification report
   * Feature importance
5. **Hyperparameter Tuning**

   * GridSearch over:

     * `n_estimators`: \[50, 100, 200]
     * `max_depth`: \[None, 10, 20]
     * `min_samples_split`: \[2, 5]
     * `min_samples_leaf`: \[1, 2]

---

## 📊 Sample Classification Report

| Class         | Precision | Recall | F1-score | Support |
| ------------- | --------- | ------ | -------- | ------- |
| 0 (Introvert) | 0.96      | 0.96   | 0.96     | 121     |
| 1 (Extrovert) | 0.94      | 0.94   | 0.94     | 79      |
| **Accuracy**  |           |        | **0.95** | 200     |

---

## 🌲 Feature Importance

Top features identified by Random Forest:

```
1. Time_spent_Alone
2. Friends_circle_size
3. Stage_fear
4. Post_frequency
5. Drained_after_socializing
```

---

## 🔧 Hyperparameter Tuning Insights

GridSearchCV was performed on 4 key parameters. Here’s how accuracy changes:

### 🔹 `n_estimators` (Number of trees)

More trees = better generalization, but longer training.

### 🔹 `max_depth`

Controls how deep each tree can go. None = unlimited.

### 🔹 `min_samples_split` & `min_samples_leaf`

Helps prevent overfitting. Higher values = simpler trees.

---

## 🗘️ Heatmap of GridSearch Results

| n\_estimators → / max\_depth ↓ | 10    | 20    | None  |
| ------------------------------ | ----- | ----- | ----- |
| 50                             | 0.939 | 0.936 | 0.936 |
| 100                            | 0.938 | 0.936 | 0.936 |
| 200                            | 0.938 | 0.936 | 0.936 |

*(CV Accuracy)*

---

## 🚀 How to Run (on Google Colab)

1. Open `Google Colab`
2. Upload the dataset
3. Paste the full notebook code provided in this repo
4. Run cells sequentially

---

## 🧠 Next Steps

* Add SHAP explainability for local/global feature impact
* Try other classifiers (e.g., SVM, XGBoost)
* Deploy model using Streamlit or Flask

---

Code: https://colab.research.google.com/drive/1K96GZDWekVVYaMcq5iJvKjz8nB4f5f2f?usp=sharing


## 📜 License

MIT License © 2025
