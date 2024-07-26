#Importando Bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importando o Dataset
dataset = pd.read_csv('Data.csv') #Leia o arquivo CSV
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Imprimindo o Dataset
x,y

#Tratando os dados NaN
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
imputer.transform(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

#Imprimindo o Dataset atualizado
print(x)

#Variavel Independente
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
ct.fit_transform(x)
x = np.array(ct.fit_transform(x))

#Variavel Dependente
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
