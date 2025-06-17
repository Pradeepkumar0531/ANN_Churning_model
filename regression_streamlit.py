import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import numpy as np
import pandas as pd
import pickle

# Loading the trained model
model=tf.keras.models.load_model('regression_model.keras', compile=True)

# Loading the Encoders and scaling
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

# Streamlit app
st.title('Customer Salary Prediction')

#User input
Geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
Gender = st.selectbox('Gender',label_encoder_gender.classes_)
Age = st.slider('Age',18,92)
Balance = st.number_input('Balance')
Credit_score = st.number_input('Credit Score')
Tenure = st.slider('Tenure',0,10)
Number_of_products = st.slider('Number of Products',1,4)
Has_credit_card = st.selectbox('Has Credit Card',[0,1])
Is_activer_mentor = st.selectbox('Is Active Member',[0,1])
Exited = st.selectbox('Exited',[0,1])

# Preparing the input data
input_data={
    'CreditScore':[Credit_score],
    'Gender': [label_encoder_gender.transform([Gender])[0]],
    'Age': [Age],
    'Tenure': [Tenure],
    'Balance': [Balance],
    'NumOfProducts': [Number_of_products],
    'HasCrCard': [Has_credit_card],
    'IsActiveMember': [Is_activer_mentor],
    'Exited': [Exited]
}

# One Hot Encode 'Geography'
geo_encoded=onehot_encoder_geo.transform([[Geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data=pd.DataFrame(input_data)
# Combining ohe columns into main dataframe
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# Predict on button click
if st.button("Predict Salary"):
    expected_columns = [
        'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance',
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited',
        'Geography_France', 'Geography_Germany', 'Geography_Spain'
    ]
    input_data = input_data[expected_columns]  # Ensure correct column order

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    prediction_salary = prediction[0][0]
    st.success(f'Predicted Estimated Salary: ₹ {prediction_salary:,.2f}')
