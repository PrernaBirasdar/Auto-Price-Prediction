🚗 Auto Price Prediction System
📌 Project Overview

This project focuses on predicting car prices using various machine learning models. The dataset required extensive preprocessing due to missing values, categorical inconsistencies, and outliers. After experimentation, an Artificial Neural Network (ANN) emerged as the best-performing model with an R² Score of 0.9714 and RMSE of 1409.53.

⚡ Model Performance Insights

Linear Regression performed poorly (R² = -40) due to multicollinearity.

Ensemble models (Random Forest, Gradient Boosting, XGBoost) showed strong performance.

ANN outperformed all models, making it the final choice.

🛠️ Challenges Faced During the Project
1. Data Cleaning Challenges

Challenge:

Raw dataset contained missing values marked as "?" instead of NaN.

Numeric columns (e.g., normalized-losses, bore, stroke, horsepower, peak-rpm) were stored as object types.

Solution:

Replaced "?" with np.nan.

Converted object columns to numeric using pd.to_numeric(errors='coerce').

Filled missing numeric values with median and categorical with mode.

2. Handling Categorical Data

Challenge:

Columns like num-of-cylinders and num-of-doors were stored as words ('two', 'four', etc.).

Solution:

Mapped words to numbers via dictionary mapping.

Verified mappings using value_counts().

3. Outlier Detection and Treatment

Challenge:

Key numeric columns (normalized-losses, engine-size, horsepower, price, etc.) contained extreme outliers.

Solution:

Visualized outliers using boxplots.

Applied IQR capping method (Q1 - 1.5*IQR, Q3 + 1.5*IQR).

Verified distributions after treatment.

4. Feature Engineering

Challenge:

Proper feature scaling was required for distance-based models (KNN, SVM) and ANN.

Solution:

Applied StandardScaler on continuous features.

Saved scaler object with joblib for deployment consistency.

5. Model Creation & Inconsistent Outputs

Challenge:

Models gave inconsistent results due to random initialization and data splitting.

Solution:

Fixed random_state=42 across all splits and estimators.

For ANN:

np.random.seed(42)
tf.random.set_seed(42)

6. Model Evaluation & Performance Issues

Challenge:

Linear Regression gave negative R² and was unsuitable.

Solution:

Removed Linear Regression.

Focused on Random Forest, Gradient Boosting, XGBoost, and ANN.

Selected ANN as the best model.

✅ Summary

Through effective data preprocessing, feature engineering, and model evaluation, the project successfully built an Auto Price Prediction System.

Best Model: Artificial Neural Network (ANN)

Performance: R² = 0.9714, RMSE = 1409.53

📂 Tech Stack

Python (Pandas, NumPy, Scikit-learn, TensorFlow, XGBoost)

Visualization: Matplotlib, Seaborn

Modeling: Linear Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, ANN

🚀 Future Enhancements

Deploy model as a Flask/Django web app.

Add a streamlit dashboard for interactive predictions.

Experiment with Hyperparameter Optimization (Optuna/RandomizedSearchCV).
