from flask import Flask, request, render_template
import pickle
import numpy as np
from Train import ModeloML
from Train import tratar_dados_novos

# Carregue seu modelo de ML
with open('trained_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Receba os dados do formulário
    features = [float(x) for x in request.form.values()]
    final_features = features
    final_features_transform = tratar_dados_novos(dados=final_features)
    
    # Faça a previsão usando o modelo
    prediction = model.predict(final_features_transform)
    if prediction[0] == 1:
        diabetes='Diabético'
        perc = prediction[1]
    else:
        diabetes='Não Diabético'
        perc = 1-prediction[1]
    
    return render_template('index.html', prediction_text=f'A previsão do modelo é {perc*100}% de ser {diabetes}')

if __name__ == "__main__":
    app.run(debug=True)