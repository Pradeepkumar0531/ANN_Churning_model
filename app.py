import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import numpy as np
import pandas as pd
import pickle

# Loading the trained model
model=tf.keras.models.load_model('model.keras', compile=False)

# Loading the Encoders and scaling
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

# Streamlit app
st.title('Customer Churn Prediction')

#User input
Geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
Gender = st.selectbox('Gender',label_encoder_gender.classes_)
Age = st.slider('Age',18,92)
Balance = st.number_input('Balance')
Credit_score = st.number_input('Credit Score')
Estimated_Salary = st.number_input('Estimated Salary')
Tenure = st.slider('Tenure',0,10)
Number_of_products = st.slider('Number of Products',1,4)
Has_credit_card = st.selectbox('Has Credit Card',[0,1])
Is_activer_mentor = st.selectbox('Is Active Member',[0,1])

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
    'EstimatedSalary': [Estimated_Salary]
}

# One Hot Encode 'Geography'
geo_encoded=onehot_encoder_geo.transform([[Geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data=pd.DataFrame(input_data)
# Combining ohe columns into main dataframe
input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

# Scale the input data
input_scaled = scaler.transform(input_data)

# Prediction
prediction=model.predict(input_scaled)
prediction_proba=prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.warning('The customer is **likely to churn**.')
else:
    st.success('The customer is **not likely to churn**.')
