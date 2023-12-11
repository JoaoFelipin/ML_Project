# Modelo de previsão de chance de diabetes

Este código define uma classe chamada `ModeloML` que é usada para treinar e testar um modelo de Regressão Logística.

## Parte 1: Inicialização do Modelo

O código começa definindo a classe `ModeloML`. O construtor da classe (`__init__`) recebe um caminho para um arquivo CSV, lê os dados desse arquivo e os armazena em `self.dados`.

## Parte 2: Pré-processamento, Treinamento e Teste

A classe `ModeloML` contém os seguintes métodos:

- `preprocess(self, coluna_alvo, norm_col, norm_is_true=False)`: Este método é usado para pré-processar os dados. Ele separa a variável alvo dos dados, divide os dados em conjuntos de treinamento e teste e normaliza os dados se `norm_is_true` for `True`.

- `treinar_modelo(self)`: Este método é usado para treinar o modelo de Regressão Logística nos dados de treinamento.

- `testar_modelo(self)`: Este método é usado para testar o modelo nos dados de teste. Ele retorna a acurácia do modelo e a matriz de confusão.

## Parte Final: Previsões e Execução

A classe `ModeloML` também contém um método para fazer previsões:

- `predict(self, novos_dados)`: Este método é usado para fazer previsões em novos dados. Ele retorna as previsões e as probabilidades das previsões.

Além disso, há uma função `tratar_dados_novos(dados)` definida fora da classe que é usada para tratar novos dados antes de fazer previsões. Ela recebe uma lista de dados, transforma em um array numpy, remodela o array e, em seguida, transforma em um DataFrame pandas.

Finalmente, há uma função `run()` que cria um objeto `ModeloML`, pré-processa os dados, treina o modelo, testa o modelo e faz previsões em novos dados.

Para executar o código, basta chamar a função `run()`.
