import pandas as pd
import numpy as np
import pickle

class healthInsurance:
    def __init__(self):
        self.home_path = ''
        self.annual_premium_scaler = pickle.load(open ('models/annual_premium_scaler.pkl', 'rb'))
        self.age_scaler = pickle.load(open ('models/age_scaler.pkl', 'rb'))
        self.vintage_scaler = pickle.load(open ('models/vintage_scaler.pkl', 'rb'))
        self.gender_scaler = pickle.load(open ('models/gender_scaler.pkl', 'rb'))
        self.region_code_scaler = pickle.load(open ('models/region_code_scaler.pkl', 'rb'))
        self.policy_sales_scaler = pickle.load(open ('models/policy_sales_scaler.pkl', 'rb'))
    
    def data_cleaning (self, df1):     
        return df1
    
    def feature_engineering (self, df2):
        df2['vehicle_damage'] = df2['vehicle_damage'].apply(lambda x: 1 if x == 'Yes' else 0)
        return df2
    
    def data_preparation(self, df4):
        
                
        cols_selected = ['vintage', 'annual_premium', 'age',
                        'region_code', 'vehicle_damage',
                        'policy_sales_channel', 'previously_insured']
        return df4[ cols_selected]
    
    def get_prediction (self, model, original_data, test_data):
        
        pred = model.predict_proba(test_data)
        
        original_data['score'] = pred[:, 1].tolist()
        
        return original_data.to_json(orient='records', date_format='iso')