import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

df = pd.read_csv("penguins_raw.csv")

df.drop('Unnamed: 0', axis=1, inplace=True)

df2 = pd.DataFrame()
df2['Culmen Length (mm)']  = df['Culmen Length (mm)']
df2['Culmen Depth (mm)']  = df['Culmen Depth (mm)']
df2['Flipper Length (mm)']  = df['Flipper Length (mm)']
df2['Body Mass (g)']  =  df['Body Mass (g)']
df2.dropna(inplace=True)

print(df2)

X = df2[0]#,df2[1],df2[2]
y = df2[3]

# X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LinearRegression
# linreg = LinearRegression()
# linreg.fit(X_train, y_train)
print('sucesso')
# l1 = linreg.score(X_train, y_train)
# l2 = linreg.score(X_test, y_test)
# print("R² of Linear Regression on training set: {:.3f}".format(l1))
# print("R² of Linear Regression on test set: {:.3f}".format(l2))