import pickle
import streamlit as st 
import pandas as pd 


st.set_page_config(page_title='Deploy do modelo de diabetes')
st.title('Diabetes Prediction')

#parametros
idade = st.number_input(label='High BP',value=18,min_value=18,max_value=120)
bmi = st.number_input(label='HighChol',value=30.0)
children = st.slider(label='CholCheck',min_value=0)
smoker = st.selectbox(label='BMI',options=[0,1])
smoker = st.selectbox(label='Smoker',options=[0,1])

with open('models/model.pkl','rb') as model_file:
    model = pickle.load(model_file)

def Prediction():
    df_input = pd.DataFrame([{
        'age':idade,
        'bmi':bmi,
        'children':children,
        'smoker':smoker

    }])
    pred = model.predict(df_input)[0]

    return pred

insurance_cost = Prediction()

st.write(insurance_cost)