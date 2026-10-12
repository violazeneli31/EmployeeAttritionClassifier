# 💼 Employee Attrition Predictor

A machine learning web app that predicts whether an employee is likely to leave a company, built with **Streamlit**, **XGBoost** and **Random Forest**.

## How we built it

1. **Dataset:** We used the IBM HR Analytics Employee Attrition dataset: 1,470 employees with 35 attributes such as age, income, overtime, job satisfaction, work-life balance and years at the company. Only about 16% of employees in it left, so the data is heavily imbalanced.
2. **Preprocessing:** Categorical columns (department, job role, marital status, etc.) were label-encoded, and all features were scaled with a `RobustScaler`, which is less sensitive to outliers such as very high salaries.
3. **Feature engineering:** We created 10 new features that capture relationships between attributes, for example *overtime × job satisfaction*, *distance from home × overtime*, *income per year of experience* and *years since last promotion relative to total experience*.
4. **Handling imbalance:** The data was split 80/20 into training and test sets (keeping the same attrition ratio in both). On the training set we applied **SMOTE-Tomek**, which creates synthetic examples of employees who left and removes borderline cases, taking the ratio from 5.2 : 1 to 1 : 1.
5. **Training and tuning:** We trained a **Random Forest** and an **XGBoost** model, testing 10 hyperparameter combinations for each, and tuned the decision threshold to get the best F1 score.
6. **Model selection:** XGBoost performed best (F1 = 0.56 vs. 0.55 for Random Forest) and is the recommended model in the app, but both are available for comparison.
7. **Business thresholds:** Instead of a single yes/no cutoff, the app offers three strategies so HR can choose between catching more at-risk employees or raising fewer false alarms.
8. **Web app:** The trained models were saved and loaded into a Streamlit app where you can test example employees, enter your own data, and explore the model's performance.

## Features

- **Example profiles:** 10 real employees from the test set, with their actual outcome and the model's prediction
- **Manual input:** enter any employee's details and get an attrition risk score
- **Model choice:** XGBoost or Random Forest
- **Threshold strategies:** High Recall, Balanced or High Precision
- **Performance metrics:** accuracy, F1, ROC-AUC, confusion matrices, classification reports and feature importance

## Results (test set)

| Model         | Accuracy | F1    | ROC-AUC |
|---------------|----------|-------|---------|
| XGBoost       | 86.7%    | 0.562 | 0.809   |
| Random Forest | 80.6%    | 0.551 | 0.803   |

XGBoost with different thresholds:

| Strategy       | Threshold | Precision | Recall |
|----------------|-----------|-----------|--------|
| High Recall    | 0.10      | 0.35      | 0.74   |
| Balanced       | 0.33      | 0.60      | 0.53   |
| High Precision | 0.60      | 0.71      | 0.32   |

## What's in each folder

- **`main.py`**: the Streamlit app
- **`dataset/`**: the HR employee attrition dataset (CSV)
- **`model/`**: the training notebook (`AI_Project_Code.ipynb`) and the saved models, scaler and encoders used by the app
- **`documentation/`**: the project proposal, final report and presentation (Word, PowerPoint and PDF)

## Run it locally

You need **Python 3.11 or newer**.

```bash
# 1. Check your Python version
python3 --version
# If it's older than 3.11, check whether you have a newer one installed,
# e.g. python3.12 --version, and use that name instead of python3 below.

# 2. Download the project
git clone https://github.com/violazeneli31/EmployeeAttritionClassifier.git
cd EmployeeAttritionClassifier

# 3. Create a virtual environment and install the libraries
python3 -m venv .venv            # or: python3.12 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 4. Start the app
streamlit run main.py
```

The app opens at http://localhost:8501.

**Mac users:** XGBoost needs the OpenMP library, which macOS doesn't include. If you get an error saying `libomp.dylib` could not be loaded, install it with [Homebrew](https://brew.sh) and start the app again:

```bash
brew install libomp
```

> The library versions in `requirements.txt` are pinned to the ones the models were trained with. Other versions of scikit-learn or XGBoost may fail to load the saved models.

## Dataset

[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (Kaggle)

## Authors

Sidrit Rrustemi and Viola Zeneli: CEN 352 Artificial Intelligence, SWE 3C
