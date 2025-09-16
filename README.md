# 🚗 Auto Price Prediction System

## 📖 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset Details](#-dataset-details)
3. [Project Workflow](#-project-workflow)
4. [Challenges Faced & Solutions](#-challenges-faced--solutions)
5. [Modeling & Performance](#-modeling--performance)
6. [Tech Stack](#-tech-stack)
7. [How to Run the Project](#-how-to-run-the-project)
8. [Future Enhancements](#-future-enhancements)
9. [Summary](#-summary)

---

## 📌 Project Overview

The goal of this project is to **predict car prices** based on their specifications using machine learning models.

The project involves:

* **Data Preprocessing** (handling missing values, categorical encoding, outliers).
* **Exploratory Data Analysis (EDA)** for insights.
* **Feature Engineering** (scaling, transformations).
* **Model Training & Evaluation** using multiple algorithms.
* Selection of the **best model** based on performance metrics.

The final system is capable of predicting auto prices with **high accuracy**.

---

## 📊 Dataset Details

* **Source**: [UCI Machine Learning Repository – Automobile Dataset](https://archive.ics.uci.edu/ml/datasets/Automobile)
* **Size**: 205 rows × 26 columns
* **Target Variable**: `price`

### Key Features:

* **Categorical**: `make`, `fuel-type`, `aspiration`, `num-of-doors`, `body-style`, `drive-wheels`, `engine-location`, `num-of-cylinders`, `fuel-system`
* **Numerical**: `wheel-base`, `length`, `width`, `height`, `curb-weight`, `engine-size`, `bore`, `stroke`, `compression-ratio`, `horsepower`, `peak-rpm`, `city-mpg`, `highway-mpg`

---

## 🛠️ Project Workflow

1. **Data Cleaning** – handled missing values & type conversions.
2. **EDA** – statistical analysis & visualizations.
3. **Feature Engineering** – encoding, scaling, outlier treatment.
4. **Model Building** – trained multiple ML models.
5. **Model Evaluation** – compared models with RMSE, R² score.
6. **Final Model Selection** – ANN chosen as best model.

---

## ⚡ Challenges Faced & Solutions

### 1. Data Cleaning Challenges

* **Problem**: Missing values marked as `"?"` and numerical columns stored as `object`.
* **Solution**:

  * Replaced `"?"` with `np.nan`.
  * Converted columns using `pd.to_numeric(errors='coerce')`.
  * Filled missing numerical values with **median** and categorical with **mode**.

---

### 2. Handling Categorical Data

* **Problem**: Textual values like `'two'`, `'four'` in `num-of-doors` and `num-of-cylinders`.
* **Solution**: Created **mapping dictionaries** to convert words into numeric values.

---

### 3. Outlier Detection & Treatment

* **Problem**: Extreme values in `normalized-losses`, `engine-size`, `horsepower`, `price`.
* **Solution**:

  * Visualized using **boxplots**.
  * Applied **IQR capping** method.
  * Verified distributions after treatment.

---

### 4. Feature Engineering

* **Problem**: Models like **KNN, SVM, ANN** require scaled data.
* **Solution**:

  * Applied **StandardScaler** on continuous features.
  * Saved scaler with `joblib` for deployment.

---

### 5. Model Creation & Inconsistent Outputs

* **Problem**: Random results due to different splits and initialization.
* **Solution**:

  * Fixed `random_state=42`.
  * Set seeds for reproducibility:

    ```python
    np.random.seed(42)
    tf.random.set_seed(42)
    ```

---

### 6. Model Evaluation Issues

* **Problem**: Linear Regression gave **R² = -40** (very poor).
* **Solution**:

  * Removed Linear Regression.
  * Focused on **Ensemble models & ANN**.

---

## 📈 Modeling & Performance

### Models Tested:

* Linear Regression ❌ (poor performance)
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost
* KNN
* SVM
* **Artificial Neural Network (ANN)** ✅

### Final Results:

* **Best Model**: ANN
* **R² Score**: 0.9714
* **RMSE**: 1409.53

---

## 💻 Tech Stack

* **Programming**: Python
* **Libraries**: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow, XGBoost
* **Tools**: Jupyter Notebook, Joblib

---


## 🚀 Future Enhancements

* Deploy using **Flask/Django** web app.
* Build an **interactive Streamlit dashboard**.
* Apply **Hyperparameter Tuning** with Optuna or RandomizedSearchCV.
* Try **Deep Learning (CNN/LSTM)** architectures for improvement.

---

## ✅ Summary

This project demonstrates how **careful preprocessing, feature engineering, and model selection** can produce a high-performing predictive system.

* **Final Model**: Artificial Neural Network (ANN)
* **Performance**: R² = 0.9714 | RMSE = 1409.53
* Successfully built an **Auto Price Prediction System** capable of assisting in car valuation.

---

✨ If you found this project helpful, don’t forget to ⭐ the repository!

---
