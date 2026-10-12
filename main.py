import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score

# Keep sklearn from cluttering our output with warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Build file paths relative to this script, so the app works no matter
# which folder you launch `streamlit run` from (locally or on Streamlit Cloud)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
DATA_PATH = BASE_DIR / "dataset" / "HR-Employee-Attrition.csv"

# -----------------------------------------
# PAGE SETUP
# -----------------------------------------
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="💼",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .risk-yes {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .risk-no {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        font-size: 1.1rem;
    }
    .example-card {
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
        cursor: pointer;
        background-color: white;
    }
    .example-card:hover {
        border-color: #1f77b4;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .example-card.selected {
        border-color: #1f77b4;
        background-color: #f0f7ff;
        box-shadow: 0 4px 12px rgba(31,119,180,0.2);
    }
    .stayed-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .left-badge {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .misclassified-badge {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.85rem;
        margin-left: 0.5rem;
    }
    .prediction-panel {
        position: sticky;
        top: 20px;
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
    }
    .employee-list {
        max-height: 800px;
        overflow-y: auto;
        padding-right: 0.5rem;
    }
    .employee-list::-webkit-scrollbar {
        width: 8px;
    }
    .employee-list::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }
    .employee-list::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 10px;
    }
    .employee-list::-webkit-scrollbar-thumb:hover {
        background: #555;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">💼 Employee Attrition Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict employee turnover risk using machine learning • Optimize retention strategies</div>', unsafe_allow_html=True)

# -----------------------------------------
# LOAD MODELS AND PREPROCESSING ARTIFACTS
# -----------------------------------------
@st.cache_resource
def load_rf_model():
    """Load the Random Forest model we trained earlier"""
    try:
        with open(MODEL_DIR / "final_rf_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("⚠️ Random Forest model file not found. Please check the path: model/final_rf_model.pkl")
        return None

@st.cache_resource
def load_xgb_model():
    """Load the XGBoost model - usually performs a bit better than RF"""
    try:
        with open(MODEL_DIR / "final_xgb_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("⚠️ XGBoost model file not found. Please check the path: model/final_xgb_model.pkl")
        return None

@st.cache_resource
def load_scaler():
    """Load the scaler we fit during training - super important to use the SAME one!"""
    try:
        with open(MODEL_DIR / "scaler.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("⚠️ Scaler file not found. Please check the path: model/scaler.pkl")
        return None

@st.cache_resource
def load_label_encoders():
    """Load the label encoders for categorical features - must match training exactly"""
    try:
        with open(MODEL_DIR / "label_encoders.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("⚠️ Label encoders file not found. Please check the path: model/label_encoders.pkl")
        return None

@st.cache_resource
def load_feature_order():
    """Load the exact order of features from training - models are picky about this!"""
    try:
        with open(MODEL_DIR / "feature_order.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("⚠️ Feature order file not found. Please check the path: model/feature_order.pkl")
        return None

# Load all the artifacts we need
rf_model = load_rf_model()
xgb_model = load_xgb_model()
scaler = load_scaler()
label_encoders = load_label_encoders()
feature_order = load_feature_order()

# Make sure everything loaded successfully before continuing
if None in [rf_model, xgb_model, scaler, label_encoders, feature_order]:
    st.error("⚠️ Critical files missing. Please ensure all model artifacts are saved:")
    st.code("""
        # In your training notebook, save these:
        with open('model/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
            
        with open('model/label_encoders.pkl', 'wb') as f:
            pickle.dump(label_encoders, f)
            
        with open('model/feature_order.pkl', 'wb') as f:
            pickle.dump(X.columns.tolist(), f)
            """)
    st.stop()

# -----------------------------------------
# LOAD DATASET FOR REFERENCE (UI only)
# -----------------------------------------
@st.cache_data
def load_dataset():
    """Load the original dataset for UI purposes (dropdowns, examples, etc.)"""
    try:
        df = pd.read_csv(DATA_PATH)
        # Only keep valid attrition values
        df = df[df['Attrition'].isin(['Yes','No'])]
        return df
    except FileNotFoundError:
        st.error("⚠️ Dataset file not found. Please check the path: dataset/HR-Employee-Attrition.csv")
        return None

dataset = load_dataset()

if dataset is None:
    st.stop()

# Pre-configured employee examples for testing
EXAMPLES = {
    "Employee 1": {
        "actual": "No",
        "description": "44-year-old experienced researcher with excellent work-life balance and tenure",
        "Age": 44,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 1117,
        "Department": "Research & Development",
        "DistanceFromHome": 2,
        "Education": 1,
        "EducationField": "Life Sciences",
        "EmployeeCount": 1,
        "EmployeeNumber": 1246,
        "EnvironmentSatisfaction": 1,
        "Gender": "Female",
        "HourlyRate": 72,
        "JobInvolvement": 4,
        "JobLevel": 1,
        "JobRole": "Research Scientist",
        "JobSatisfaction": 4,
        "MaritalStatus": "Married",
        "MonthlyIncome": 2011,
        "MonthlyRate": 19982,
        "NumCompaniesWorked": 1,
        "Over18": "Y",
        "OverTime": "No",
        "PercentSalaryHike": 13,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 4,
        "StandardHours": 80,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 10,
        "TrainingTimesLastYear": 5,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 10,
        "YearsInCurrentRole": 5,
        "YearsSinceLastPromotion": 7,
        "YearsWithCurrManager": 7
    },
    "Employee 2": {
        "actual": "No",
        "description": "35-year-old research scientist with strong engagement and excellent training",
        "Age": 35,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 1370,
        "Department": "Research & Development",
        "DistanceFromHome": 27,
        "Education": 4,
        "EducationField": "Life Sciences",
        "EmployeeCount": 1,
        "EmployeeNumber": 1728,
        "EnvironmentSatisfaction": 4,
        "Gender": "Male",
        "HourlyRate": 49,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": "Manufacturing Director",
        "JobSatisfaction": 3,
        "MaritalStatus": "Married",
        "MonthlyIncome": 6883,
        "MonthlyRate": 5151,
        "NumCompaniesWorked": 2,
        "Over18": "Y",
        "OverTime": "No",
        "PercentSalaryHike": 16,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 2,
        "StandardHours": 80,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 17,
        "TrainingTimesLastYear": 3,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 7,
        "YearsInCurrentRole": 7,
        "YearsSinceLastPromotion": 0,
        "YearsWithCurrManager": 7
    },
    "Employee 3": {
        "actual": "No",
        "description": "35-year-old research scientist with strong tenure and work-life balance",
        "Age": 35,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 819,
        "Department": "Research & Development",
        "DistanceFromHome": 18,
        "Education": 5,
        "EducationField": "Life Sciences",
        "EmployeeCount": 1,
        "EmployeeNumber": 1621,
        "EnvironmentSatisfaction": 2,
        "Gender": "Male",
        "HourlyRate": 48,
        "JobInvolvement": 4,
        "JobLevel": 2,
        "JobRole": "Research Scientist",
        "JobSatisfaction": 1,
        "MaritalStatus": "Married",
        "MonthlyIncome": 5208,
        "MonthlyRate": 26312,
        "NumCompaniesWorked": 1,
        "Over18": "Y",
        "OverTime": "No",
        "PercentSalaryHike": 11,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 4,
        "StandardHours": 80,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 16,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 16,
        "YearsInCurrentRole": 15,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 10
    },
    "Employee 4": {
        "actual": "No",
        "description": "39-year-old research scientist with excellent work environment and high satisfaction",
        "Age": 39,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 1329,
        "Department": "Sales",
        "DistanceFromHome": 4,
        "Education": 4,
        "EducationField": "Life Sciences",
        "EmployeeCount": 1,
        "EmployeeNumber": 182,
        "EnvironmentSatisfaction": 4,
        "Gender": "Female",
        "HourlyRate": 47,
        "JobInvolvement": 2,
        "JobLevel": 2,
        "JobRole": "Sales Executive",
        "JobSatisfaction": 3,
        "MaritalStatus": "Married",
        "MonthlyIncome": 5902,
        "MonthlyRate": 14590,
        "NumCompaniesWorked": 4,
        "Over18": "Y",
        "OverTime": "No",
        "PercentSalaryHike": 14,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 3,
        "StandardHours": 80,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 17,
        "TrainingTimesLastYear": 1,
        "WorkLifeBalance": 4,
        "YearsAtCompany": 15,
        "YearsInCurrentRole": 11,
        "YearsSinceLastPromotion": 5,
        "YearsWithCurrManager": 9
    },
    "Employee 5": {
        "actual": "Yes",
        "description": "31-year-old manufacturing director with overtime, far commute, and 6-year promotion gap",
        "Age": 31,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 330,
        "Department": "Research & Development",
        "DistanceFromHome": 22,
        "Education": 4,
        "EducationField": "Medical",
        "EmployeeCount": 1,
        "EmployeeNumber": 1389,
        "EnvironmentSatisfaction": 4,
        "Gender": "Male",
        "HourlyRate": 98,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": "Manufacturing Director",
        "JobSatisfaction": 3,
        "MaritalStatus": "Married",
        "MonthlyIncome": 6179,
        "MonthlyRate": 21057,
        "NumCompaniesWorked": 1,
        "Over18": "Y",
        "OverTime": "Yes",
        "PercentSalaryHike": 15,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 4,
        "StandardHours": 80,
        "StockOptionLevel": 2,
        "TotalWorkingYears": 10,
        "TrainingTimesLastYear": 3,
        "WorkLifeBalance": 2,
        "YearsAtCompany": 10,
        "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 6,
        "YearsWithCurrManager": 7
    },
    "Employee 6": {
        "actual": "Yes",
        "description": "33-year-old researcher with very poor environment, overtime, and 9-year promotion gap",
        "Age": 33,
        "BusinessTravel": "Travel_Frequently",
        "DailyRate": 1076,
        "Department": "Research & Development",
        "DistanceFromHome": 3,
        "Education": 3,
        "EducationField": "Life Sciences",
        "EmployeeCount": 1,
        "EmployeeNumber": 702,
        "EnvironmentSatisfaction": 1,
        "Gender": "Male",
        "HourlyRate": 70,
        "JobInvolvement": 3,
        "JobLevel": 1,
        "JobRole": "Research Scientist",
        "JobSatisfaction": 1,
        "MaritalStatus": "Single",
        "MonthlyIncome": 3348,
        "MonthlyRate": 3164,
        "NumCompaniesWorked": 1,
        "Over18": "Y",
        "OverTime": "Yes",
        "PercentSalaryHike": 11,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 1,
        "StandardHours": 80,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 10,
        "TrainingTimesLastYear": 3,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 10,
        "YearsInCurrentRole": 8,
        "YearsSinceLastPromotion": 9,
        "YearsWithCurrManager": 7
    },
    "Employee 7": {
        "actual": "Yes",
        "description": "26-year-old researcher with poor environment, overtime, and job-hopping history",
        "Age": 26,
        "BusinessTravel": "Travel_Frequently",
        "DailyRate": 342,
        "Department": "Research & Development",
        "DistanceFromHome": 2,
        "Education": 3,
        "EducationField": "Life Sciences",
        "EmployeeCount": 1,
        "EmployeeNumber": 1053,
        "EnvironmentSatisfaction": 1,
        "Gender": "Male",
        "HourlyRate": 57,
        "JobInvolvement": 3,
        "JobLevel": 1,
        "JobRole": "Research Scientist",
        "JobSatisfaction": 1,
        "MaritalStatus": "Married",
        "MonthlyIncome": 2042,
        "MonthlyRate": 15346,
        "NumCompaniesWorked": 6,
        "Over18": "Y",
        "OverTime": "Yes",
        "PercentSalaryHike": 14,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 2,
        "StandardHours": 80,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 6,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 3,
        "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 2
    },
    "Employee 8": {
        "actual": "Yes",
        "description": "25-year-old sales rep with very poor environment, minimal experience, overtime, and distant location",
        "Age": 25,
        "BusinessTravel": "Travel_Frequently",
        "DailyRate": 599,
        "Department": "Sales",
        "DistanceFromHome": 24,
        "Education": 1,
        "EducationField": "Life Sciences",
        "EmployeeCount": 1,
        "EmployeeNumber": 1273,
        "EnvironmentSatisfaction": 3,
        "Gender": "Male",
        "HourlyRate": 73,
        "JobInvolvement": 1,
        "JobLevel": 1,
        "JobRole": "Sales Representative",
        "JobSatisfaction": 4,
        "MaritalStatus": "Single",
        "MonthlyIncome": 1118,
        "MonthlyRate": 8040,
        "NumCompaniesWorked": 1,
        "Over18": "Y",
        "OverTime": "Yes",
        "PercentSalaryHike": 14,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 4,
        "StandardHours": 80,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 1,
        "TrainingTimesLastYear": 4,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 1,
        "YearsInCurrentRole": 0,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 0
    },
    "Employee 9": {
        "actual": "No",
        "description": "40-year-old manufacturing director with overtime and low stock options",
        "Age": 40,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 448,
        "Department": "Research & Development",
        "DistanceFromHome": 16,
        "Education": 3,
        "EducationField": "Life Sciences",
        "EmployeeCount": 1,
        "EmployeeNumber": 1641,
        "EnvironmentSatisfaction": 3,
        "Gender": "Female",
        "HourlyRate": 84,
        "JobInvolvement": 3,
        "JobLevel": 3,
        "JobRole": "Manufacturing Director",
        "JobSatisfaction": 4,
        "MaritalStatus": "Single",
        "MonthlyIncome": 7945,
        "MonthlyRate": 19948,
        "NumCompaniesWorked": 6,
        "Over18": "Y",
        "OverTime": "Yes",
        "PercentSalaryHike": 15,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 4,
        "StandardHours": 80,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 18,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 2,
        "YearsAtCompany": 4,
        "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 3,
        "YearsWithCurrManager": 3
    },
    "Employee 10": {
        "actual": "Yes",
        "description": "25-year-old sales rep with very poor environment and low satisfaction across multiple dimensions",
        "Age": 25,
        "BusinessTravel": "Travel_Rarely",
        "DailyRate": 383,
        "Department": "Sales",
        "DistanceFromHome": 9,
        "Education": 2,
        "EducationField": "Life Sciences",
        "EmployeeCount": 1,
        "EmployeeNumber": 1439,
        "EnvironmentSatisfaction": 1,
        "Gender": "Male",
        "HourlyRate": 68,
        "JobInvolvement": 2,
        "JobLevel": 1,
        "JobRole": "Sales Representative",
        "JobSatisfaction": 1,
        "MaritalStatus": "Married",
        "MonthlyIncome": 4400,
        "MonthlyRate": 15182,
        "NumCompaniesWorked": 3,
        "Over18": "Y",
        "OverTime": "No",
        "PercentSalaryHike": 12,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 1,
        "StandardHours": 80,
        "StockOptionLevel": 0,
        "TotalWorkingYears": 6,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 3,
        "YearsAtCompany": 3,
        "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 2,
        "YearsWithCurrManager": 2
    }
}
def create_engineered_features(df):
    """
    Create all those fancy calculated features we used during training
    These help the model understand relationships between variables
    """
    df = df.copy()
    
    # How long have they been in their current role compared to total experience?
    df['YearsInRole_Ratio'] = df['YearsInCurrentRole'] / (df['TotalWorkingYears'] + 1)
    
    # Interaction between overtime and satisfaction (both matter a lot!)
    df['OverTime_Satisfaction'] = df['OverTime'] * df['JobSatisfaction']
    
    # Income relative to experience level
    df['Income_Per_Year'] = df['MonthlyIncome'] / (df['TotalWorkingYears'] + 1)
    
    # How often do they get promoted relative to experience?
    df['Promotion_Rate'] = df['YearsSinceLastPromotion'] / (df['TotalWorkingYears'] + 1)
    
    # How much of their career has been at this company?
    df['Tenure_Ratio'] = df['YearsAtCompany'] / (df['TotalWorkingYears'] + 1)
    
    # Combined work-life and satisfaction score
    df['Work_Life_Balance'] = df['WorkLifeBalance'] * df['JobSatisfaction']
    
    # Long commute + overtime = recipe for burnout
    df['Distance_OverTime'] = df['DistanceFromHome'] * df['OverTime']
    
    # Income relative to job level (are they paid fairly for their level?)
    df['Income_JobLevel'] = df['MonthlyIncome'] / (df['JobLevel'] + 1)
    
    # Experience relative to age
    df['Experience_Age_Ratio'] = df['TotalWorkingYears'] / (df['Age'] + 1)
    
    # Total training investment
    df['Training_Years'] = df['TrainingTimesLastYear'] * df['YearsAtCompany']
    
    return df

def make_prediction(input_data, model_choice, threshold):
    """
    Take employee data and predict if they're at risk of leaving
    This handles all the preprocessing to match what we did during training
    """
    if xgb_model is None or rf_model is None or scaler is None or label_encoders is None or feature_order is None:
        st.error("Models or preprocessing artifacts not loaded. Cannot make predictions.")
        return None, None
    
    # Convert input dictionary to a DataFrame
    input_df = pd.DataFrame([{k: v for k, v in input_data.items() 
                              if k not in ['description', 'actual', 'misclassified']}])
    
    # Make sure all numeric columns are actually numeric (not strings!)
    numeric_features = [
        'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount',
        'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
        'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate',
        'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating',
        'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
        'YearsWithCurrManager'
    ]
    
    for col in numeric_features:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    
    # Convert categorical features to numbers using our saved encoders
    for col, le in label_encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = input_df[col].astype(str)
                input_df[col] = le.transform(input_df[col])
            except ValueError as e:
                st.error(f"Error encoding {col}: {e}")
                st.error(f"Value: {input_df[col].iloc[0]}, Valid values: {list(le.classes_)}")
                return None, None
    
    # Add all those engineered features
    input_df = create_engineered_features(input_df)
    
    # Make sure columns are in the exact same order as training
    try:
        input_df = input_df[feature_order]
    except KeyError as e:
        st.error(f"Missing features: {e}")
        return None, None
    
    # Scale the features using our saved scaler
    input_df_scaled = scaler.transform(input_df)
    
    # Get the prediction from whichever model they chose
    model = xgb_model if model_choice == "XGBoost" else rf_model
    prob = float(model.predict_proba(input_df_scaled)[0][1])
    prediction = 1 if prob > threshold else 0
    
    return prob, prediction

def display_results(prob, prediction, threshold, model_choice, actual_outcome=None):
    """Show the prediction results in a nice, easy-to-read format"""
    if prob is None:
        return
    
    # Figure out what to display
    predicted_label = "Yes (Left)" if prediction == 1 else "No (Stayed)"
    risk_class = "risk-yes" if prediction == 1 else "risk-no"
    icon = "⚠️" if prediction == 1 else "✅"
    
    # Big, colorful prediction card
    st.markdown(f'<div class="metric-box {risk_class}"><h2>{icon} {predicted_label}</h2><p style="font-size: 2rem; margin: 1rem 0;">{prob*100:.1f}% Probability</p></div>', unsafe_allow_html=True)
    
    # Details section
    st.markdown("### 📊 Prediction Details")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Threshold:** {threshold:.2%}")
        st.markdown(f"**Model:** {model_choice}")
    
    with col2:
        if actual_outcome:
            actual_text = "Yes (Left)" if actual_outcome == "Yes" else "No (Stayed)"
            
            # Check if we got it right
            if (actual_outcome == "Yes" and prediction == 1) or (actual_outcome == "No" and prediction == 0):
                st.success(f"✅ Correct: {actual_text}")
            else:
                st.warning(f"⚠️ Misclassified\nActual: {actual_text}")
    
    # Explain the prediction
    if prediction == 1:
        st.error(f"Risk score ({prob:.1%}) exceeds threshold ({threshold:.1%})")
    else:
        st.success(f"Risk score ({prob:.1%}) is below threshold ({threshold:.1%})")

# -----------------------------------------
# TABS FOR DIFFERENT SECTIONS
# -----------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📊 Example Profiles", "✏️ Manual Input", "📈 Performance Metrics", "ℹ️ About"])

with tab1:
    st.markdown("### 🚀 Employee Attrition Analysis")
    st.markdown("Select an employee from the list to analyze their attrition risk. The prediction will appear below the employee card.")
    
    # Initialize session state
    if 'selected_example' not in st.session_state:
        st.session_state.selected_example = list(EXAMPLES.keys())[0]
    
    # Prediction settings at the top
    st.markdown("### ⚙️ Prediction Settings")
    st.markdown("Choose your model and threshold strategy before selecting an employee.")
    
    col1, col2 = st.columns(2)
    with col1:
        model_choice_ex = st.selectbox("Model", ["XGBoost", "Random Forest"], key="model_ex")
        st.session_state.model_choice_ex = model_choice_ex
    with col2:
        threshold_choice_ex = st.selectbox("Threshold Strategy", ["High Recall", "Balanced", "High Precision"], index=1, key="thresh_ex")
        st.session_state.threshold_choice_ex = threshold_choice_ex
    
    threshold_map = {"High Recall": 0.10, "Balanced": 0.33, "High Precision": 0.60}
    threshold_ex = threshold_map[threshold_choice_ex]
    
    st.markdown("---")
    
    # Single column layout for employees
    st.markdown("#### 👥 Employee Profiles")
    
    # Display employees in a single column with predictions below when selected
    for name, data in EXAMPLES.items():
        is_selected = (st.session_state.selected_example == name)
        badge_text = "No (Stayed)" if data['actual'] == "No" else "Yes (Left)"
        card_class = "example-card selected" if is_selected else "example-card"
        
        st.markdown(f"""
        <div class="{card_class}">
            <strong>{name}</strong>
            <br><small style="color: #666;">{data['description']}</small>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(f"Select", key=f"btn_{name}", width='stretch'):
            st.session_state.selected_example = name
            st.rerun()
        
        # Show prediction immediately below if this employee is selected
        if is_selected:
            st.markdown("---")
            st.markdown(f"### 📋 {name}")
            
            # Display key attributes in a compact grid
            st.markdown("#### Key Attributes")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Age", f"{data['Age']} yrs")
                st.metric("Tenure", f"{data['YearsAtCompany']} yrs")
            
            with col2:
                st.metric("Income", f"${data['MonthlyIncome']:,}")
                st.metric("Job Level", data['JobLevel'])
            
            with col3:
                st.metric("Satisfaction", f"{data['JobSatisfaction']}/4")
                st.metric("Work-Life", f"{data['WorkLifeBalance']}/4")
            
            with col4:
                st.metric("Overtime", data['OverTime'])
                st.metric("Distance", f"{data['DistanceFromHome']} km")
            
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            # Make prediction with current settings
            prob, prediction = make_prediction(data, model_choice_ex, threshold_ex)
            
            if prob is not None:
                # Determine outcome labels
                predicted_label = "Yes (Left)" if prediction == 1 else "No (Stayed)"
                risk_class = "risk-yes" if prediction == 1 else "risk-no"
                icon = "⚠️" if prediction == 1 else "✅"
                
                # Main prediction card
                st.markdown(f'<div class="metric-box {risk_class}"><h2>{icon} {predicted_label}</h2><p style="font-size: 2rem; margin: 1rem 0;">{prob*100:.1f}% Probability</p></div>', unsafe_allow_html=True)
                
                # Prediction details
                st.markdown("#### 📊 Prediction Details")
                st.progress(prob, text=f"Attrition Risk: {prob*100:.1f}%")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Threshold:** {threshold_ex:.2%}")
                    st.markdown(f"**Model:** {model_choice_ex}")
                
                with col2:
                    actual_text = "Yes (Left)" if data['actual'] == "Yes" else "No (Stayed)"
                    
                    if (data['actual'] == "Yes" and prediction == 1) or (data['actual'] == "No" and prediction == 0):
                        st.success(f"✅ Correct: {actual_text}")
                    else:
                        st.warning(f"⚠️ Misclassified\nActual: {actual_text}")
                
                if prediction == 1:
                    st.error(f"Risk score ({prob:.1%}) exceeds threshold ({threshold_ex:.1%})")
                else:
                    st.success(f"Risk score ({prob:.1%}) is below threshold ({threshold_ex:.1%})")
            
            st.markdown("---")
    
    # Threshold comparison at the bottom
    st.markdown("---")
    st.markdown("### 🎚️ Compare Different Thresholds")
    st.markdown("---")
    st.info("""
            **Threshold Guide:**
            - **High Recall (0.10)**: Catch more at-risk employees, accept more false alarms
            - **Balanced (0.33)**: Optimal F1 score - good general purpose
            - **High Precision (0.60)**: Only flag high-confidence cases, fewer false alarms
            """)

with tab2:
    st.markdown("### ✏️ Enter Employee Details Manually")
    st.markdown("Fill in the employee information below. Focus on key factors like **satisfaction**, **overtime**, **income**, and **tenure**.")
    
    user_input = {}
    
    # Organize inputs into logical sections
    st.markdown("#### 👤 Demographics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        user_input['Age'] = st.slider("Age", 18, 60, 33, help="Employee age in years")
        user_input['Gender'] = st.selectbox("Gender", dataset['Gender'].unique().tolist())
    
    with col2:
        user_input['MaritalStatus'] = st.selectbox("Marital Status", dataset['MaritalStatus'].unique().tolist())
        user_input['DistanceFromHome'] = st.slider("Distance From Home (km)", 1, 29, 5, help="Distance between home and office")
    
    with col3:
        user_input['Education'] = st.select_slider("Education Level", options=[1, 2, 3, 4, 5], value=3, 
                                                    help="1=Below College, 2=College, 3=Bachelor, 4=Master, 5=Doctor")
        user_input['EducationField'] = st.selectbox("Education Field", dataset['EducationField'].unique().tolist())
    
    st.markdown("#### 💼 Job Details")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        user_input['Department'] = st.selectbox("Department", dataset['Department'].unique().tolist())
        user_input['JobRole'] = st.selectbox("Job Role", dataset['JobRole'].unique().tolist())
        user_input['JobLevel'] = st.select_slider("Job Level", options=[1, 2, 3, 4, 5], value=2)
    
    with col2:
        user_input['BusinessTravel'] = st.selectbox("Business Travel", dataset['BusinessTravel'].unique().tolist())
        user_input['OverTime'] = st.selectbox("Works Overtime?", ['No', 'Yes'], help="Critical factor in attrition")
        user_input['StockOptionLevel'] = st.select_slider("Stock Option Level", options=[0, 1, 2, 3], value=1)
    
    with col3:
        user_input['MonthlyIncome'] = st.number_input("Monthly Income ($)", 1000, 20000, 5000, step=500)
        user_input['PercentSalaryHike'] = st.slider("Last Salary Hike (%)", 11, 25, 15)
        user_input['NumCompaniesWorked'] = st.slider("Number of Previous Companies", 0, 9, 2)
    
    st.markdown("#### 😊 Satisfaction & Engagement (1=Low, 4=High)")
    st.markdown("**These are critical factors in predicting attrition!**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        user_input['JobSatisfaction'] = st.select_slider("Job Satisfaction", options=[1, 2, 3, 4], value=3, 
                                                         help="Major predictor of attrition")
    
    with col2:
        user_input['WorkLifeBalance'] = st.select_slider("Work-Life Balance", options=[1, 2, 3, 4], value=3,
                                                         help="Important for retention")
    
    with col3:
        user_input['EnvironmentSatisfaction'] = st.select_slider("Environment Satisfaction", options=[1, 2, 3, 4], value=3)
    
    with col4:
        user_input['JobInvolvement'] = st.select_slider("Job Involvement", options=[1, 2, 3, 4], value=3)
    
    st.markdown("#### 📅 Experience & Tenure")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        user_input['TotalWorkingYears'] = st.slider("Total Working Years", 0, 40, 10)
    
    with col2:
        user_input['YearsAtCompany'] = st.slider("Years at Company", 0, 40, 5)
    
    with col3:
        user_input['YearsInCurrentRole'] = st.slider("Years in Current Role", 0, 18, 3)
    
    with col4:
        user_input['YearsSinceLastPromotion'] = st.slider("Years Since Last Promotion", 0, 15, 1,
                                                          help="Long time without promotion may indicate dissatisfaction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        user_input['YearsWithCurrManager'] = st.slider("Years with Current Manager", 0, 17, 2)
    
    with col2:
        user_input['TrainingTimesLastYear'] = st.slider("Training Times Last Year", 0, 6, 3)
    
    with col3:
        user_input['RelationshipSatisfaction'] = st.select_slider("Relationship Satisfaction", options=[1, 2, 3, 4], value=3)
    
    # Hidden/constant fields that don't need user input
    user_input['DailyRate'] = int(dataset['DailyRate'].median())
    user_input['EmployeeCount'] = 1
    user_input['EmployeeNumber'] = int(dataset['EmployeeNumber'].median())
    user_input['HourlyRate'] = int(dataset['HourlyRate'].median())
    user_input['MonthlyRate'] = int(dataset['MonthlyRate'].median())
    user_input['Over18'] = 'Y'
    user_input['PerformanceRating'] = 3
    user_input['StandardHours'] = 80
    
    # Model and threshold selection
    st.markdown("---")
    st.markdown("### 🤖 Prediction Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_choice_manual = st.selectbox("Select Model", ["XGBoost", "Random Forest"], key="model_manual",
                                          help="XGBoost typically performs slightly better")
    
    with col2:
        threshold_choice_manual = st.selectbox("Threshold Strategy", ["High Recall", "Balanced", "High Precision"], 
                                              index=1, key="thresh_manual",
                                              help="High Recall: Catch more at-risk employees | Balanced: Optimal F1 | High Precision: Fewer false alarms")
    
    threshold_map = {"High Recall": 0.10, "Balanced": 0.33, "High Precision": 0.60}
    threshold_manual = threshold_map[threshold_choice_manual]
    
    if st.button("🔮 Predict Attrition Risk", width='stretch', key="predict_manual", type="primary"):
        prob, prediction = make_prediction(user_input, model_choice_manual, threshold_manual)
        display_results(prob, prediction, threshold_manual, model_choice_manual)

with tab3:
    st.markdown("## 📈 Model Performance Metrics")
    st.markdown("Compare how well each model performs on the test dataset")
    
    # Try to load test data if available
    try:
        with open(MODEL_DIR / "test_predictions.pkl", "rb") as f:
            test_data = pickle.load(f)
        
        y_test = test_data['y_test']
        rf_predictions = test_data['rf_predictions']
        xgb_predictions = test_data['xgb_predictions']
        rf_proba = test_data['rf_proba']
        xgb_proba = test_data['xgb_proba']
        
        st.success("✅ Test data loaded successfully!")
        
        # Model comparison metrics
        st.markdown("### 🔬 Model Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🌳 Random Forest")
            rf_acc = test_data['rf_accuracy']
            rf_f1 = f1_score(y_test, rf_predictions)
            rf_auc = roc_auc_score(y_test, rf_proba)
            
            st.metric("RF Accuracy", f"{rf_acc:.2%}")
            st.metric("F1 Score", f"{rf_f1:.3f}")
            st.metric("ROC-AUC Score", f"{rf_auc:.3f}")
            
        with col2:
            st.markdown("#### 🚀 XGBoost")
            xgb_acc = test_data['xgb_accuracy']
            xgb_f1 = f1_score(y_test, xgb_predictions)
            xgb_auc = roc_auc_score(y_test, xgb_proba)
            
            st.metric("XGB Accuracy", f"{xgb_acc:.2%}")
            st.metric("F1 Score", f"{xgb_f1:.3f}")
            st.metric("ROC-AUC Score", f"{xgb_auc:.3f}")
        
        # Confusion matrices side by side
        st.markdown("---")
        st.markdown("### 📊 Confusion Matrices")
        st.markdown("See how many predictions each model got right vs wrong")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Random Forest")
            rf_cm = confusion_matrix(y_test, rf_predictions)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Stayed', 'Left'],
                       yticklabels=['Stayed', 'Left'], ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Random Forest Confusion Matrix')
            st.pyplot(fig)
            plt.close()
            
        with col2:
            st.markdown("#### XGBoost")
            xgb_cm = confusion_matrix(y_test, xgb_predictions)
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Greens',
                       xticklabels=['Stayed', 'Left'],
                       yticklabels=['Stayed', 'Left'], ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('XGBoost Confusion Matrix')
            st.pyplot(fig)
            plt.close()
        
        # Classification reports
        st.markdown("---")
        st.markdown("### 📋 Detailed Classification Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Random Forest")
            rf_report = classification_report(y_test, rf_predictions, 
                                             target_names=['Stayed', 'Left'],
                                             output_dict=True)
            st.dataframe(pd.DataFrame(rf_report).transpose().round(3))
            
        with col2:
            st.markdown("#### XGBoost")
            xgb_report = classification_report(y_test, xgb_predictions,
                                              target_names=['Stayed', 'Left'],
                                              output_dict=True)
            st.dataframe(pd.DataFrame(xgb_report).transpose().round(3))
        
        # Feature importance comparison
        st.markdown("---")
        st.markdown("### 🎯 Feature Importance Comparison")
        st.markdown("Which features matter most for predicting attrition?")
        
        # Get feature importances
        feature_importance = pd.DataFrame({
            'Feature': feature_order,
            'RF_Importance': rf_model.feature_importances_,
            'XGB_Importance': xgb_model.feature_importances_
        })
        
        # Calculate average importance
        feature_importance['Avg_Importance'] = (
            feature_importance['RF_Importance'] + 
            feature_importance['XGB_Importance']
        ) / 2
        
        # Sort by average importance
        feature_importance = feature_importance.sort_values('Avg_Importance', ascending=False)
        
        # Show top 15 in a table
        st.markdown("#### Top 15 Most Important Features")
        top_15_features = feature_importance.head(15).copy()
        top_15_features['RF_Importance'] = top_15_features['RF_Importance'].round(4)
        top_15_features['XGB_Importance'] = top_15_features['XGB_Importance'].round(4)
        top_15_features['Avg_Importance'] = top_15_features['Avg_Importance'].round(4)
        st.dataframe(top_15_features, width='stretch', hide_index=True)
        
        # Plot comparison
        st.markdown("#### Visual Comparison")
        fig, ax = plt.subplots(figsize=(12, 8))
        
        top_features = feature_importance.head(15)
        x_pos = np.arange(len(top_features))
        width = 0.35
        
        # Create horizontal bars
        ax.barh(x_pos - width/2, top_features['RF_Importance'], 
               width, label='Random Forest', alpha=0.8, color='#3498db')
        ax.barh(x_pos + width/2, top_features['XGB_Importance'], 
               width, label='XGBoost', alpha=0.8, color='#2ecc71')
        
        ax.set_yticks(x_pos)
        ax.set_yticklabels(top_features['Feature'])
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('Top 15 Feature Importance Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.invert_yaxis()  # Highest importance at top
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Key insights
        st.markdown("---")
        st.markdown("### 💡 Key Insights")
        
        top_feature = feature_importance.iloc[0]
        st.info(f"""
        **Most Important Feature:** {top_feature['Feature']}
        - Random Forest Importance: {top_feature['RF_Importance']:.4f}
        - XGBoost Importance: {top_feature['XGB_Importance']:.4f}
        - Average Importance: {top_feature['Avg_Importance']:.4f}
        
        Both models agree that this feature is the strongest predictor of employee attrition!
        """)
        
    except FileNotFoundError:
        st.warning("⚠️ Test predictions file not found. To display performance metrics, save your test predictions:")
        st.code("""
            # After training and testing your models, save predictions:
            import pickle

            test_data = {
                'y_test': y_test,
                'rf_predictions': rf_predictions,
                'xgb_predictions': xgb_predictions,
                'rf_proba': rf_proba[:, 1],  # Probability of class 1 (attrition)
                'xgb_proba': xgb_proba[:, 1]
            }

            with open('model/test_predictions.pkl', 'wb') as f:
                pickle.dump(test_data, f)
                    """, language="python")
        
        st.markdown("---")
        st.markdown("### 📊 Feature Importance (Training Data)")
        st.markdown("Even without test data, we can show which features the models learned from:")
        
        # Show feature importance from trained models
        feature_importance = pd.DataFrame({
            'Feature': feature_order,
            'RF_Importance': rf_model.feature_importances_,
            'XGB_Importance': xgb_model.feature_importances_
        })
        
        feature_importance['Avg_Importance'] = (
            feature_importance['RF_Importance'] + 
            feature_importance['XGB_Importance']
        ) / 2
        
        feature_importance = feature_importance.sort_values('Avg_Importance', ascending=False)
        
        # Display top 15
        st.markdown("#### Top 15 Most Important Features")
        top_15 = feature_importance.head(15).copy()
        top_15['RF_Importance'] = top_15['RF_Importance'].round(4)
        top_15['XGB_Importance'] = top_15['XGB_Importance'].round(4)
        top_15['Avg_Importance'] = top_15['Avg_Importance'].round(4)
        st.dataframe(top_15, width='stretch', hide_index=True)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        top_features = feature_importance.head(15)
        x_pos = np.arange(len(top_features))
        width = 0.35
        
        ax.barh(x_pos - width/2, top_features['RF_Importance'], 
               width, label='Random Forest', alpha=0.8, color='#3498db')
        ax.barh(x_pos + width/2, top_features['XGB_Importance'], 
               width, label='XGBoost', alpha=0.8, color='#2ecc71')
        
        ax.set_yticks(x_pos)
        ax.set_yticklabels(top_features['Feature'])
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('Top 15 Feature Importance Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with tab4:
    st.markdown("## ℹ️ About This Application")
    
    st.markdown("""
    ### 🎯 Purpose
    This application predicts employee attrition risk using machine learning models trained on HR data. 
    It helps HR departments identify employees who may be at risk of leaving and take proactive retention measures.
    
    ### 🤖 Models Used
    - **XGBoost**: Gradient boosting model with high accuracy (recommended)
    - **Random Forest**: Ensemble learning model for comparison
    
    Both models were trained on historical employee data with features like satisfaction scores, tenure, 
    compensation, and work-life balance indicators.
    
    ### 📊 Key Features That Impact Attrition
    Based on feature importance analysis, the most critical factors are:
    
    1. **Overtime** - Whether employee works overtime (highly predictive)
    2. **Monthly Income** - Compensation level
    3. **Years at Company** - Employee tenure
    4. **Job Satisfaction** - Self-reported satisfaction (1-4 scale)
    5. **Work-Life Balance** - Balance rating (1-4 scale)
    6. **Years Since Last Promotion** - Career growth indicator
    7. **Distance From Home** - Commute distance
    8. **Stock Option Level** - Equity compensation
    9. **Total Working Years** - Overall career experience
    10. **Age** - Employee age
    
    ### 🎚️ Threshold Strategies
    
    - **High Recall (0.10)**: Flags more employees as at-risk. Use when retention is critical and you can afford 
      to intervene with more employees, even if some aren't actually at risk.
    
    - **Balanced (0.33)**: Optimal F1 score. Good general-purpose setting that balances precision and recall.
    
    - **High Precision (0.60)**: Only flags high-confidence cases. Use when intervention resources are limited 
      and you want to focus on employees most likely to leave.
    
    ### 📚 Dataset
    
    The model was trained on the IBM HR Analytics Employee Attrition dataset containing 1,470 employee records 
    with 35 features including demographics, job details, satisfaction scores, and attrition outcomes.
    """)
    
    st.markdown("---")
    st.markdown("**Developed for HR Analytics and Workforce Planning**")
    st.markdown("**Developed by Sidrit Rrustemi and Viola Zeneli | SWE 3C**")