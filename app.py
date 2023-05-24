import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

housing_data = pd.read_csv("penguins_raw.csv")

descr = housing_data['Comments']
feature_names = housing_data['Culmen Length (mm)']
data = housing_data['Culmen Depth (mm)']
target = housing_data['Flipper Length (mm)']
df1 = pd.DataFrame(data=data)
df1.rename(columns={0: feature_names[0], 1: feature_names[1], 2: feature_names[2], 3: feature_names[3],
 4: feature_names[4], 5: feature_names[5], 6: feature_names[6], 7: feature_names[7]}, inplace=True)
df2 = pd.DataFrame(data=target)
df2.rename(columns={0: 'Flipper Length (mm)'}, inplace=True)
housing = pd.concat([df1, df2], axis=1)
print(housing.columns)

housing.head()

print('dimension of housing data: {}'.format(housing.shape))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(housing.loc[:, housing.columns != 'Flipper Length (mm)'], housing['Flipper Length (mm)'])

from sklearn.neighbors import KNeighborsRegressor
training_score = []
test_score = []
# try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
    # build the model
    knn = KNeighborsRegressor(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # record training set score
    training_score.append(knn.score(X_train, y_train))
    # record test set score
    test_score.append(knn.score(X_test, y_test))
plt.plot(neighbors_settings, training_score, label='training score')
plt.plot(neighbors_settings, test_score, label='test score')
plt.ylabel('R²')
plt.xlabel('n_neighbors')
plt.legend()
plt.savefig('teste')