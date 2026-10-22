# Credit Score Classification Pipeline

A complete machine learning pipeline built with Python to clean, preprocess, and train classification models (Random Forest, XGBoost, LightGBM) to predict customer credit scores.

## 🚀 Mini Description
An end-to-end data science pipeline for automated data cleaning, categorical feature encoding, scaling, and hyperparameter tuning using tree-based ensemble models for credit risk assessment.

## ✨ Features
* **Data Inspection & Quality Checks:** Automated missing value detection, data type checks, and duplicate removal.
* **Feature Engineering & Preprocessing:** Handles categorical conversions, label encoding, and robust numerical scaling.
* **Model Training:** Implements multiple robust classifiers including Random Forest, XGBoost, and LightGBM.
* **Evaluation Metrics:** Evaluates model performance using Accuracy, Precision, Recall, F1-Score, and Classification Reports.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Machine Learning & Preprocessing:** Scikit-Learn, XGBoost, LightGBM

## ⚙️ Installation & Usage
1. Clone or download this repository.
2. Install the required dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm
   ```
3. Place your dataset (`data_C.csv`) in the project root directory.
4. Run the script:
   ```bash
   python script.py
   ```

## 📊 Dataset Structure
The script expects a structured CSV file (`data_C.csv`) containing demographic, financial, and behavioral attributes of customers, along with target labels.
