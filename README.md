# 💼 Employee Attrition Predictor

A machine learning web app that predicts whether an employee is likely to leave a company, built with **Streamlit**, **XGBoost** and **Random Forest** on the IBM HR Analytics Employee Attrition dataset (1,470 employees, 35 features).

## Features

- **Example profiles**: 10 real employees from the test set, with their actual outcome and the model's prediction
- **Manual input**: enter any employee's details and get an attrition risk score
- **Model choice**: XGBoost or Random Forest
- **Threshold strategies**: High Recall (0.10), Balanced (0.33) or High Precision (0.60)
- **Performance metrics**: accuracy, F1, ROC-AUC, confusion matrices, classification reports and feature importance

## Results (test set)

| Model         | Accuracy | ROC-AUC |
|---------------|----------|---------|
| XGBoost       | 86.7%    | 0.809   |
| Random Forest | 80.6%    | 0.803   |

## Project structure

```
employee_attrition_classifier/
├── main.py                  # Streamlit app
├── requirements.txt
├── dataset/
│   └── HR-Employee-Attrition.csv
├── model/
│   ├── AI_Project_Code.ipynb    # training notebook (EDA, feature engineering, SMOTE, tuning)
│   ├── final_xgb_model.pkl
│   ├── final_rf_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── feature_order.pkl
│   ├── optimal_thresholds.pkl
│   └── test_predictions.pkl
└── documentation/           # proposal, report and presentation
```

## Run it locally

Requires **Python 3.11 or newer**.

```bash
git clone https://github.com/<your-username>/employee_attrition_classifier.git
cd employee_attrition_classifier

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run main.py
```

The app opens at http://localhost:8501.

> The library versions in `requirements.txt` are pinned to the ones the models were trained with. Other versions of scikit-learn or XGBoost may fail to load the saved `.pkl` files.

## Dataset

[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) (Kaggle).

## Authors

Sidrit Rrustemi and Viola Zeneli: CEN 352 Artificial Intelligence, SWE 3C
