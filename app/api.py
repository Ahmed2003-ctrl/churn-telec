from flask import Flask, jsonify, request
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)

parent_path = Path(__file__).resolve().parent.parent


def load_model():
    model_path = parent_path  + '/model/log_reg_pipeline.joblib'
    model = joblib.load(model_path)
    return model

def extract_data(data):
    gender = data.get('gender')
    SeniorCitizen = data.get('SeniorCitizen')
    Partner= data.get('Partner')
    Dependents = data.get('Dependents')
    tenure= data.get('tenure')
    PhoneService = data.get('PhoneService')
    InternetService= data.get('InternetService')
    MultipleLines= data.get('MultipleLines')
    OnlineSecurity= data.get('OnlineSecurity')
    OnlineBackup= data.get('OnlineBackup')
    DeviceProtection= data.get('DeviceProtection')
    TechSupport = data.get('TechSupport')
    StreamingTV= data.get('StreamingTV')
    StreamingMovies = data.get( 'StreamingMovies')
    Contract= data.get('Contract')
    PaperlessBilling= data.get('PaperlessBilling')
    PaymentMethod = data.get('PaymentMethod')
    MonthlyCharges= data.get('MonthlyCharges')
    TotalCharges= data.get('TotalCharges')


    data_df = pd.DataFrame({
    'gender': ['Male'],
    'SeniorCitizen': ['No'],
    'Partner': ['No'],
    'Dependents': ['No'],
    'tenure': [1],
    'PhoneService': ['No'],
    'MultipleLines': ['No'],
    'InternetService': ['Fiber optic'],
    'OnlineSecurity': ['No'],
    'OnlineBackup': ['No'],
    'DeviceProtection': ['No'],
    'TechSupport': ['No'],
    'StreamingTV': ['No'],
    'StreamingMovies': ['No'],
    'Contract': ['Two year'],
    'PaperlessBilling': ['No'],
    'PaymentMethod': ['Electronic check'],
    'MonthlyCharges': [85.0],
    'TotalCharges': [40.0]
})
    return data_df
 


@app.get('/predict')
def show():
    data = request.form
    df = extract_data(data)
    pred = load_model().predict(df)


    if pred ==1:
        return jsonify({'prediction ': 'Churn'})
    else:
        return jsonify({'prediction ': 'Not Churn '})

app.run()