#Importando Bibliotecas


import numpy as np                #Biblioteca para trabalhar com as matrizes
import matplotlib.pyplot as plt   #Gerar os gráficos
import pandas as pd               #Importar conjunto de dados

#Importando o Dataset

dataset = pd.read_csv('Data.csv') #Declarando a variável que irá carregar o Dataset
x = dataset.iloc[:, :-1].values   #Carrega as colunas da tabela, menos a última
y = dataset.iloc[:, -1].values    #Carrega somente a última coluna da tabela

print(x,y)

#Tratando dados faltantes

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #Substituir os valores em faltantes
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

print(x)

#Codando a Variavel Independente

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
ct.fit_transform(x)
x = np.array(ct.fit_transform(x))

x

#Codando a variavel dependente

from sklearn.preprocessing import LabelEncoder
#Convertendo para 0 (Não) e 1 (Sim)
le = LabelEncoder()
y = le.fit_transform(y)

y
