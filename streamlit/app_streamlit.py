import pickle
import streamlit as st 
import pandas as pd 
from sklearn import *

st.set_page_config(page_title='Deploy do modelo de diabetes')
st.title('Diabetes Prediction')

#parametros
High_BP = st.selectbox(label='High BP',options=[0,1])
high_col = st.selectbox(label='HighChol',options=[0,1])
CholCheck = st.selectbox(label='CholCheck',options=[0,1])
bmi = st.slider(label='BMI',min_value=0.0,max_value=100.0)
smoker = st.selectbox(label='Smoker',options=[0,1])
Stroke = st.selectbox(label='Stroke',options=[0,1])
Heart_Disease = st.selectbox(label='Heart Disease',options=[0,1])
Physical_Activity = st.selectbox(label='Physical Activity',options=[0,1])
Fruits = st.selectbox(label='Fruits',options=[0,1])
Veggies = st.selectbox(label='Veggies',options=[0,1])
HvyAlcohol = st.selectbox(label='HvyAlcohol',options=[0,1])
HealthCare = st.selectbox(label='HealthCare',options=[0,1])
NoDocBcCost = st.selectbox(label='NoDocBcCost',options=[0,1])
GenHealth = st.selectbox(label='GenHealth',options=[0,1,2,3,4,5])
MentHealth = st.slider(label='MentHealth',min_value=0,max_value=30)
PhysHealth = st.slider(label='PhysHealth',min_value=0,max_value=30)
DiffWalk = st.selectbox(label='DiffWalk',options=[0,1])
sex = st.selectbox(label='sex',options=[0,1])
Age = st.selectbox(label='Age',options=[0,1,2,3,4,5,6,7,8,9,10,11,12,13])
Education = st.selectbox(label='Education',options=[0,1,2,3,4,5,6])
Income = st.selectbox(label='Income',options=[0,1,2,3,4,5,6,7,8,9,10,11])


with open('trained_classifier.pkl','rb') as model_file:
    model = pickle.load(model_file)

def Prediction():
    df_input = pd.DataFrame([{
        'HighBP':High_BP,
        'HighChol':high_col,
        'CholCheck':CholCheck,
        'BMI':bmi,
        'Smoker':smoker,
        'Stroke':Stroke,
        'HeartDiseaseorAttack':Heart_Disease,
        'PhysActivity':Physical_Activity,
        'Fruits':Fruits,
        'Veggies':Veggies,
        'HvyAlcoholConsump':HvyAlcohol,
        'AnyHealthcare':HealthCare,
        'NoDocbcCost':NoDocBcCost,
        'GenHlth':GenHealth,
        'MentHlth':MentHealth,
        'PhysHlth':PhysHealth,
        'DiffWalk':DiffWalk,
        'Sex':sex,
        'Age':Age,
        'Education':Education,
        'Income':Income
        
    }])
    pred = model.predict(df_input)[0]
    perc = model.predict(df_input)[1]

    return pred,perc

diabetes_pred = Prediction()

if diabetes_pred[0] == 1:
    diabetes='Diabético'
    perc = diabetes_pred[1]
else:
    diabetes='Não Diabético'
    perc = 1-diabetes_pred[1]
    
st.write('Você tem ',perc*100,' % ',' de ser ', diabetes)