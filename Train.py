import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

class ModeloML:
    def __init__(self,path):
        self.dados = pd.read_csv(path)
        self.x = None
        self.y = None
        self.modelo = None
        
    def preprocess(self,coluna_alvo,norm_col):
        #Fazendo a separação das variaveis target
        self.x=self.dados.drop(columns=coluna_alvo,axis=1)
        self.y = self.dados[coluna_alvo]
        
        #Train/test split
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x,self.y,train_size=0.2,random_state=100)

        #Normalização dos dados
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train[norm_col])
        self.x_test = scaler.fit_transform(self.x_test[norm_col])
        
        #treinando o modelo
    def treinar_modelo(self):
        self.modelo = LogisticRegression()
        self.modelo.fit(self.x_train,self.y_train)
    
        #fazendo testes
    def testar_modelo(self):
        y_pred = self.modelo.predict(self.y_test)
        return y_pred

        #função para previsões
    def predict(self,novos_dados):
        pred = self.modelo.predict(novos_dados)
        return pred
    
    
modelo = ModeloML(path="C:/Users/anali/Downloads/diabetes_binary_5050split_health_indicators_BRFSS2021.csv")
modelo.preprocess(coluna_alvo='Diabetes_binary',norm_col=['BMI'])
modelo.treinar_modelo()


