import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix
import numpy as np

class ModeloML:
    def __init__(self,path):
        self.dados = pd.read_csv(path)
        self.x = None
        self.y = None
        self.modelo = None
        
    def preprocess(self,coluna_alvo,norm_col,norm_is_true=False):
        #Fazendo a separação das variaveis target
        self.x=self.dados.drop(columns=coluna_alvo,axis=1)
        self.y = self.dados[coluna_alvo]
        
        #Train/test split
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x,self.y,train_size=0.2,random_state=100)

        #Normalização dos dados
        if norm_is_true == True:
            scaler = StandardScaler()
            self.x_train[norm_col] = scaler.fit_transform(self.x_train[norm_col])
            self.x_test[norm_col] = scaler.fit_transform(self.x_test[norm_col])
        else:
            pass
        #treinando o modelo
    def treinar_modelo(self):
        self.modelo = LogisticRegression(max_iter=1000)
        self.modelo.fit(self.x_train,self.y_train)
    
        #fazendo testes
    def testar_modelo(self):
        y_pred = self.modelo.predict(self.x_test)
        acc = accuracy_score(y_pred=y_pred,y_true=self.y_test)
        c_matrix = confusion_matrix(y_pred=y_pred,y_true=self.y_test)
        return acc,c_matrix

        #função para previsões
    def predict(self,novos_dados):
        pred = self.modelo.predict(X=novos_dados)
        pred_prob = self.modelo.predict_proba(novos_dados)
        return pred,pred_prob[0][1]
    
def tratar_dados_novos(dados):
    array = np.array(dados).reshape(1,-1)
    df = pd.DataFrame(array,columns=['HighBP','HighChol','CholCheck','BMI','Smoker','Stroke','HeartDiseaseorAttack','PhysActivity','Fruits','Veggies','HvyAlcoholConsump','AnyHealthcare','NoDocbcCost','GenHlth','MentHlth','PhysHlth','DiffWalk','Sex','Age','Education','Income'])
    return df

def run(novos_dados):
    modelo = ModeloML(path="C:/Users/anali/Downloads/diabetes_binary_5050split_health_indicators_BRFSS2021.csv")
    modelo.preprocess(coluna_alvo='Diabetes_binary',norm_col=['BMI'],norm_is_true=False)
    modelo.treinar_modelo()
    print(modelo.testar_modelo())
    print(modelo.predict(novos_dados=tratar_dados_novos(novos_dados)))

run(novos_dados=[0,0.0,0,35.0,0.0,0.0,0.0,0,0,0,0,0,0.0,1.0,0.0,0.0,0.0,1,3,6.0,11.0])