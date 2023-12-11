import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

class ModeloML:
    def __init__(self,path):
        self.dados = pd.read_csv("C:/Users/anali/Downloads/diabetes_binary_5050split_health_indicators_BRFSS2021.csv")
    
    def preprocess(self,coluna_alvo):
        #Fazendo a separação das variaveis target
        self.x=self.dados.drop(columns=coluna_alvo,axis=1)
        self.y = self.dados[coluna_alvo]
        
        #Train/test split
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x,self.y,train_size=0.2,random_state=100)

        #Normalização dos dados
        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.fit_transform(self.x_test)



