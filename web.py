import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Load model
model=pickle.load(open('LinearRegressionModel.pkl','rb'))

# Load data
df = pd.read_csv('Clean_moto_data.csv')

def predict_price(model, input_data):
    # Preprocess input data
    input_features = ['model_name', 'company', 'model_year', 'kms_driven', 'mileage', 'power', 'engine']
    input_df = pd.DataFrame(data=[input_data], columns=input_features)

    # Make prediction
    predicted_price = model.predict(input_df)

    return predicted_price[0]

def main():
    # Set page title
    st.title('Motorbike Price Predictor')

    # Add input fields for features
    model_name = st.selectbox('Model Name', df['model_name'].unique())
    company = st.selectbox('Company', df['company'].unique())
    model_year = st.number_input('Model Year', min_value=1900, max_value=2023, value=2021)
    kms_driven = st.number_input('Kilometers Driven', min_value=0, value=1000)
    mileage = st.number_input('Mileage (kmpl)', min_value=0.0, value=50.0, step=0.1)
    power = st.number_input('Power (bhp)', min_value=0, value=50)
    engine = st.number_input('Engine (cc)', min_value=0, value=100)

    # Create a dictionary with the input values
    input_data = {
        'model_name': model_name,
        'company': company,
        'model_year': model_year,
        'kms_driven': kms_driven,
        'mileage': mileage,
        'power': power,
        'engine': engine
    }

    # Predict the price
    if st.button('Predict Price'):
        predicted_price = predict_price(model, input_data)
        st.success(f'Predicted Price: {predicted_price:.2f} Lakh')

if __name__ == '__main__':
    main()

#streamlit run web.py