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
        news_cols = ['id', 'gender', 'age', 'driving_license', 'region_code',
       'previously_insured', 'vehicle_age', 'vehicle_damage', 'annual_premium',
       'policy_sales_channel', 'vintage', 'response']

        df1.columns = news_cols
        return df1
    
    def feature_engineering (self, df2):
        df2['vehicle_age'] = df2['vehicle_age'].apply(lambda x: 'over_2_years' if x== '> 2 Years'
                                            else 'between_1_2_year' if x == '1-2 Year'
                                            else 'below_1_year' )

        df2['vehicle_damage'] = df2['vehicle_damage'].apply(lambda x: 1 if x == 'Yes' else 0)
        return df2
    
    def data_preparation(self, df4):
        # df4['annual_premium'] = self.annual_premium_scaler.transform(df4[['annual_premium']].values)
        
        # df4['age'] = self.age_scaler.transform(df4[['age']].values)
        # df4['vintage'] = self.vintage_scaler.transform(df4[['vintage']].values)
        
        # df4.loc[:, 'gender'] = df4['gender'].map(self.gender_scaler)
        
        # df4.loc[:, 'region_code'] = df4['region_code'].map(self.region_code_scaler)
          
        # df4 = pd.get_dummies(df4, prefix='vehicle_age', columns=['vehicle_age'], dtype=float)
                
        # df4.loc[:, 'policy_sales_channel'] = df4['policy_sales_channel'].map(self.policy_sales_scaler)
        df4['annual_premium'] = df4['annual_premium'].astype(float)
        df4['region_code'] = df4['region_code'].astype(int)
        df4['policy_sales_channel'] = df4['policy_sales_channel'].astype(int)
        
        cols_selected = ['vintage', 'annual_premium', 'age',
                        'region_code', 'vehicle_damage',
                        'policy_sales_channel', 'previously_insured']
        return df4[ cols_selected]
    
    def get_prediction (self, model, original_data, test_data):
        
        pred = model.predict_proba(test_data)
        
        original_data['score'] = pred[:, 1].tolist()
        
        return original_data.to_json(orient='records', date_format='iso')