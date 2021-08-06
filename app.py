import _pickle as pickle
#import data viz and manipulation/cleaning libraries/EDA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme(color_codes=True)
import base64

#import machine learning libraries
from sklearn.model_selection import train_test_split, KFold 
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import Normalizer, StandardScaler, RobustScaler

#streamlit
import streamlit as st

from pathlib import Path

df_cat = pd.read_csv("co2_emissions.csv") 

def main():
    st.set_page_config(page_title='t3nt3n_ONE - Linear Regression App', page_icon="🖖")
    st.title('Automotive C02 Emissions Predictor')
    st.sidebar.title("User Inputs For Model")
    st.sidebar.markdown("Car Features")

if __name__ == '__main__':
    main()

padding = 0
st.markdown(f""" <style>
    .reportview-container .main .block-container{{
        padding-top: {padding}rem;
        padding-right: {padding}rem;
        padding-left: {padding}rem;
        padding-bottom: {padding}rem;
    }} </style> """, unsafe_allow_html=True)
st.write("""

# Multi Linear Regression App

 Instructions: 
 Select inputs from sidebar to see model predictions on CO2 (g/km) emissions for automobiles.
 Keys and values for categorical features are under the dropdown menu selection

Model uses multi linear regression with data scaled using StandardScaler (Z-Score)

""")

df_selected = pd.read_csv('co2_emissions_encoded.csv')

df_selected_all = df_selected[['make', 'vehicle_class', 'engine_liters', 'cylinders', 'transmission',
       'fuel_type', 'fuel_consumption_liters_100km', 'co2_emissions',
       'co2_bins']].copy()


def filedownload(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="co2_emissions_encoded.csv">Download this csv dataset</a>'

st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(filedownload(df_selected_all), unsafe_allow_html=True)


def user_input_features():
        make = st.sidebar.selectbox('Car Brand',(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39))
        #st.sidebar.write("Keys and Values Dictionary to Decode for Car Brand Inputs:")
        st.sidebar.write(pd.DataFrame({
    'Keys & Values': ['0: acura','1: alfa romeo', '2: aston martin', '3: audi','4: bmw','5: buick','6: cadillac','7: chevrolet','8: chrysler','9: dodge','10: fiat','11: ford','12: genesis','13: gmc','14: honda','15: hyundai','16: infiniti','17: jaguar','18: jeep','19: kia','20: lamborghini','21: land rover','22: lexus','23: lincoln','24: maserati','25: mazda','26: mercedes-benz','27: mini','28: mitsubishi','29: nissan','30: porsche','31: ram','32: rolls-royce','33: subaru','34: toyota','35: volkswagen','36: volvo','37: bentley','38: bugatti'],
}))
        vehicle_class = st.sidebar.selectbox('Vehicle Class',(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14))
        st.sidebar.write(pd.DataFrame({
    'Keys & Values': ['0: suv: small','1: two-seater','2: compact','3: mid-size','4: subcompact','5: minicompact','6: station wagon: small','7: full-size','8: suv: standard','9: pickup truck: small','10: pickup truck: standard','11: minivan','12: van: passenger','13: special purpose vehicle','14: station wagon: mid-size'],    
}))
        engine_liters = st.sidebar.slider('Engine Size (L)', 1.0,10.0, 3.0)
        cylinders = st.sidebar.slider('Cylinders', 2.0,16.0, 6.0)
        transmission = st.sidebar.selectbox('Transmission',(0,1,2,3,4))
        st.sidebar.write(pd.DataFrame({
    'Keys & Values': ['0: Automatic with Select Shift','1: Automated Manual','2: Automatic','3: Manual','4: Continuously Variable',],    
}))
        fuel_type = st.sidebar.selectbox('Fuel Type',(0,1,2,3))
        st.sidebar.write(pd.DataFrame({
    'Keys & Values': ['0: Premium Gasoline','1: Regular Gasoline','2: Diesel','3: Ethanol(E85)'],    
}))

        fuel_consumption = st.sidebar.slider('Fuel Consumption', 2.0,27.0, 10.6)

  
        data = {'make':[make],
                'vehicle_class':[vehicle_class],
                'engine_liters':[engine_liters],
                'cylinders':[cylinders],
                'transmission':[transmission],
                'fuel_type':[fuel_type],
                'fuel_consumption_liters_100km':[fuel_consumption]
                }

        features = pd.DataFrame(data)

        return features

input_df = user_input_features()

co2_raw = pd.read_csv('co2_emissions_encoded.csv')
co2_raw.fillna(0, inplace=True)
co2 = co2_raw.drop(columns=['co2_emissions', 'co2_bins'])
df = pd.concat([input_df,co2],axis=0)

#will have to label encode here
for col in df:
    if df[col].dtype == 'object':
        key, value = pd.factorize(df[col]) 

df = df[:1] # Selects only the first row (the user input data)
df.fillna(0, inplace=True)

df = df[:1] # Selects only the first row (the user input data)
df.fillna(0, inplace=True)

features = ['make', 'vehicle_class', 'engine_liters', 'cylinders', 'transmission',
       'fuel_type', 'fuel_consumption_liters_100km']

df = df[features]


# Displays the user input features
st.subheader('User Input features')
print(df.columns)
st.write(df)

# Reads in saved classification model
load_mlr = pickle.load(open('multi_linear.pkl', 'rb'))

# Apply model to make predictions
prediction = load_mlr.predict(df)


#Write the output
st.subheader('Predicted CO2 Emissions (g/km)')

st.write(prediction)


"""


  
  
  """