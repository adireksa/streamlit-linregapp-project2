import _pickle as pickle
#import data viz and manipulation/cleaning libraries/EDA
import pandas as pd
import numpy as np

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
        #make selectbox
        display_make = ('0: acura','1: alfa romeo', '2: aston martin', '3: audi','4: bmw','5: buick','6: cadillac','7: chevrolet','8: chrysler','9: dodge','10: fiat','11: ford','12: genesis','13: gmc','14: honda','15: hyundai','16: infiniti','17: jaguar','18: jeep','19: kia','20: lamborghini','21: land rover','22: lexus','23: lincoln','24: maserati','25: mazda','26: mercedes-benz','27: mini','28: mitsubishi','29: nissan','30: porsche','31: ram','32: rolls-royce','33: subaru','34: toyota','35: volkswagen','36: volvo','37: bentley','38: bugatti')
        options_make = list(range(len(display_make)))
        make = st.sidebar.selectbox("Car Brand", options_make, format_func=lambda x: display_make[x])

        #vehicle_class selectbox
        display_vehicle_class = ('0: suv: small','1: two-seater','2: compact','3: mid-size','4: subcompact','5: minicompact','6: station wagon: small','7: full-size','8: suv: standard','9: pickup truck: small','10: pickup truck: standard','11: minivan','12: van: passenger','13: special purpose vehicle','14: station wagon: mid-size')
        options_vehicle_class = list(range(len(display_vehicle_class)))
        vehicle_class = st.sidebar.selectbox("Vehicle Class", options_vehicle_class, format_func=lambda x: display_vehicle_class[x])

        engine_liters = st.sidebar.slider('Engine Size (L)', 1.0,10.0, 3.0)

        cylinders = st.sidebar.slider('Cylinders', 2.0,16.0, 6.0)

        #transmission selectbox
        display_transmission = ('0: Automatic with Select Shift','1: Automated Manual','2: Automatic','3: Manual','4: Continuously Variable')
        options_transmission = list(range(len(display_transmission)))
        transmission = st.sidebar.selectbox("Transmission", options_transmission, format_func=lambda x: display_transmission[x])
        #fuel type selectbox
        display_fuel_type = ('0: Premium Gasoline','1: Regular Gasoline','2: Diesel','3: Ethanol(E85)')
        options_fuel_type = list(range(len(display_fuel_type)))
        fuel_type = st.sidebar.selectbox("Fuel Type", options_fuel_type, format_func=lambda x: display_fuel_type[x])

        fuel_consumption = st.sidebar.slider('Fuel Consumption Combined (L/100 km)', 2.0,27.0, 10.6)

  
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


#insert image to see quartiles of target
st.image('co2_quartiles.png')




#st.sidebar.write(transmission_sidebar)
"""
        
        
        
  
  
  """