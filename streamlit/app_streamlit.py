import pickle
import streamlit as st 
import pandas as pd 


st.set_page_config(page_title='Deploy do modelo de diabetes')
st.title('Diabetes Prediction')

#parametros
idade = st.number_input(label='High BP',value=18,min_value=18,max_value=120)
high_col = st.number_input(label='HighChol',value=30.0)
CholCheck = st.slider(label='CholCheck',min_value=0)
bmi = st.slider(label='BMI',min_value=0)
smoker = st.selectbox(label='Smoker',options=[0,1])
smoker = st.selectbox(label='Stroke',options=[0,1])
smoker = st.selectbox(label='Heart Disease',options=[0,1])
smoker = st.selectbox(label='Physical Activity',options=[0,1])
smoker = st.selectbox(label='Fruits',options=[0,1])
smoker = st.selectbox(label='Veggies',options=[0,1])
smoker = st.selectbox(label='HvyAlcohol',options=[0,1])
smoker = st.selectbox(label='HealthCare',options=[0,1])
smoker = st.selectbox(label='NoDocBcCost',options=[0,1])
smoker = st.selectbox(label='GenHealth',options=[0,1])
smoker = st.selectbox(label='MentHealth',options=[0,1])
smoker = st.selectbox(label='PhysHealth',options=[0,1])
smoker = st.selectbox(label='DiffWalk',options=[0,1])
smoker = st.selectbox(label='sex',options=[0,1])
smoker = st.selectbox(label='Age',options=[0,1])
smoker = st.selectbox(label='Education',options=[0,1])
smoker = st.selectbox(label='Income',options=[0,1])


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