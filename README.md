# Projeto Python IA: Inteligência Artificial e Previsões

## Case: Score de Crédito dos Clientes

Este projeto tem como objetivo desenvolver um modelo de **Inteligência Artificial (IA)** para prever o score de crédito dos clientes de um banco. O score de crédito será categorizado em Ruim, Ok ou Bom com base nas informações dos clientes.

### Passos do Projeto

#### Passo 1 - Importar a base de dados

Para iniciar o projeto, é necessário importar a base de dados dos clientes do banco.

import pandas as pd

tabela = pd.read_csv("clientes.csv")

#### Passo 2 - Preparar a base de dados para a IA

Neste passo, realizamos o pré-processamento dos dados para que a IA possa entender. Isso inclui codificar colunas de texto em números utilizando o `LabelEncoder` do scikit-learn.

from sklearn.preprocessing import LabelEncoder

codificador = LabelEncoder()

# Exemplo de codificação para coluna 'profissao'
tabela["profissao"] = codificador.fit_transform(tabela["profissao"])

#### Passo 3 - Criar um modelo de IA

Foram utilizados dois modelos diferentes para treinar a IA: **RandomForest** e **K-Nearest Neighbors (KNN)**.

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Criando modelos
modelo_arvoredecisao = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()

# Treinando os modelos
modelo_arvoredecisao.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)

#### Passo 4 - Escolher o melhor modelo

Neste passo, comparamos a precisão dos modelos treinados para escolher o melhor.

from sklearn.metrics import accuracy_score

# Fazendo previsões e calculando a precisão
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste)

accuracy_arvoredecisao = accuracy_score(y_teste, previsao_arvoredecisao)
accuracy_knn = accuracy_score(y_teste, previsao_knn)

print(f"Precisão Árvore de Decisão: {accuracy_arvoredecisao:.2f}")
print(f"Precisão KNN: {accuracy_knn:.2f}")

# Considerações Finais

Este projeto demonstra como desenvolver e implementar um modelo de IA para previsão de score de crédito dos clientes utilizando Python e bibliotecas como pandas e scikit-learn. A escolha do modelo ideal depende da precisão alcançada após avaliação e comparação entre diferentes algoritmos.
