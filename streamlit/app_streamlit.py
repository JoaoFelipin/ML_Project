import pickle
import streamlit as st 
import pandas as pd 
import Train

st.set_page_config(page_title='Deploy do modelo de diabetes')
st.title('Diabetes Prediction')

#parametros
High_BP = st.number_input(label='High BP',value=18,min_value=18,max_value=120)
high_col = st.number_input(label='HighChol',value=30.0)
CholCheck = st.slider(label='CholCheck',min_value=0)
bmi = st.slider(label='BMI',min_value=0)
smoker = st.selectbox(label='Smoker',options=[0,1])
Stroke = st.selectbox(label='Stroke',options=[0,1])
Heart_Disease = st.selectbox(label='Heart Disease',options=[0,1])
Physical_Activity = st.selectbox(label='Physical Activity',options=[0,1])
Fruits = st.selectbox(label='Fruits',options=[0,1])
Veggies = st.selectbox(label='Veggies',options=[0,1])
HvyAlcohol = st.selectbox(label='HvyAlcohol',options=[0,1])
HealthCare = st.selectbox(label='HealthCare',options=[0,1])
NoDocBcCost = st.selectbox(label='NoDocBcCost',options=[0,1])
GenHealth = st.selectbox(label='GenHealth',options=[0,1])
MentHealth = st.selectbox(label='MentHealth',options=[0,1])
PhysHealth = st.selectbox(label='PhysHealth',options=[0,1])
DiffWalk = st.selectbox(label='DiffWalk',options=[0,1])
sex = st.selectbox(label='sex',options=[0,1])
Age = st.selectbox(label='Age',options=[0,1])
Education = st.selectbox(label='Education',options=[0,1])
Income = st.selectbox(label='Income',options=[0,1])


with open('trained_classifier.pkl','rb') as model_file:
    model = pickle.load(model_file)

def Prediction():
    df_input = pd.DataFrame([{
        'age':idade,
        'bmi':bmi,
        'smoker':smoker

    }])
    pred = model.predict(df_input)[0]

    return pred

insurance_cost = Prediction()

st.write(insurance_cost)