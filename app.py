import streamlit as st 
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler , LabelEncoder, OneHotEncoder
import pickle

model = tf.keras.models.load_model('model.h5')

with open('label_encode_gender.pkl', 'rb') as file:
    label_encode_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


st.title("Customer Churn Prediction")
geography = st.selectbox('Geography',onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encode_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input("Balance")
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider("Tenser", 1, 10)
num_of_Product = st.slider("Number of Products", 1,4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_activate_member = st.selectbox("Is Activated Member",[0,1])


input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encode_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_Product],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_activate_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoder_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoder_df], axis=1)

input_data_sacaled = scaler.transform(input_data)

prediction = model.predict(input_data_sacaled)
prediction_proba = prediction[0][0]
st.write(f"Churn Probability:{prediction_proba:.2f}")

if prediction_proba >0.5:
    st.write("The Customer likely to churn")
else:
    st.write("The Customer is not likely to churn")
