import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Definindo DataFrame
df = pd.read_csv("penguins_raw.csv")

# Drop valores com zero
df.drop('Unnamed: 0', axis=1, inplace=True)

# Definindo DataFrame formatado
df2 = pd.DataFrame()
df2['Culmen Length (mm)']  = df['Culmen Length (mm)']
df2['Culmen Depth (mm)']  = df['Culmen Depth (mm)']
df2['Body Mass (g)']  =  df['Body Mass (g)']
df2['Flipper Length (mm)']  = df['Flipper Length (mm)']
# Drop de valores NaN
df2.dropna(inplace=True)

# Definindo y e x
y = df2['Flipper Length (mm)']
X = df2.drop(columns=['Flipper Length (mm)'])

# Treinando algoritmo
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Testando valores
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)
# print('sucesso')
l1 = linreg.score(X_train, y_train)
l2 = linreg.score(X_test, y_test)
print("R² de Linear Regression de teste: {:.3f}".format(l1))
print("R² de Linear Regression de treino: {:.3f}".format(l2))