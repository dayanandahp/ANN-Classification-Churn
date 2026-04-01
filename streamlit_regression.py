import streamlit as st 
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
import pandas as pd
import pickle

model = tf.keras.models.load_model('regression_model.h5')

with open('reg_label_encoder_gender.pkl','rb') as file:
    reg_label_encoder_gender = pickle.load(file)
with open('reg_onehot_encoder_geo.pkl', 'rb') as file:
    reg_onehot_encoder_geo = pickle.load(file)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


st.title("Estimated Salary Prediction")
# 'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
#        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
#        'Exited'

geography = st.selectbox('Geography',reg_onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender',reg_label_encoder_gender.classes_)
age = st.slider("Age",18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input("Credit Score")
tenure = st.slider("Tenure", 0,10)
exited = st.selectbox("Exited", [0,1])
num_of_products = st.slider("Number of products",1,4)
has_cr_card =st.selectbox("Has credit card",[0,1])
is_active_member = st.selectbox("Is Activate Member",[0,1])


input_data = pd. DataFrame({
'CreditScore': [credit_score],
'Gender':[reg_label_encoder_gender.transform([gender])[0]],
'Age': [age],
'Tenure': [tenure],
'Balance': [balance],
'NumOfProducts': [num_of_products],
'HasCrCard': [has_cr_card],
'IsActiveMember': [is_active_member],
'Exited': [exited]})


geo_encoded = reg_onehot_encoder_geo.transform(pd.DataFrame({'Geography':[geography]})).toarray()
geo_encoder_df = pd.DataFrame(geo_encoded, columns=reg_onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoder_df], axis=1)

input_data_sacaled = scaler.transform(input_data)

prediction = model.predict(input_data_sacaled)
prediction_proba = prediction[0][0]
st.write(f"Predicated Estimated salry: ${prediction_proba:.2f}")

 
