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
    
    return render_template('index.html', prediction_text=f'A previsão do modelo é {prediction[1]*100}% de ser {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)