import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics

df = pd.read_csv("penguins_raw.csv")

df.drop('Unnamed: 0', axis=1, inplace=True)

df2 = pd.DataFrame()
df2['Culmen Length (mm)']  = df['Culmen Length (mm)']
df2['Culmen Depth (mm)']  = df['Culmen Depth (mm)']
df2['Flipper Length (mm)']  = df['Flipper Length (mm)']
df2['Body Mass (g)']  =df['Body Mass (g)']

print(df2)
df2.dropna(inplace=True)
print(df2)

X = df2[0],df2[1],df2[2]
y = df2[3]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
# print(X_train.shape)
# print(X_test.shape)

# print(y_train.shape)
# print(y_test.shape)

# X_train = X_train.values
# X_train = X_train.reshape(-1, 1)

# X_test = X_test.values
# X_test = X_test.reshape(-1, 1)

# # print(X_train)
# # print(X_test)

# y_train = y_train.values
# y_train = y_train.reshape(-1, 1)

# y_test = y_test.values
# y_test = y_test.reshape(-1, 1)

# print(y_train)
# print(y_test)

k_range = range(1,11)
scores = {}
scores_list = {}
for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores[k] = metrics.accuracy_score  (y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test,y_pred))
