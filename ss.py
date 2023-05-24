import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
housing_data = fetch_california_housing()

descr = housing_data[‘DESCR’]
feature_names = housing_data[‘feature_names’]
data = housing_data[‘data’]
target = housing_data[‘target’]
df1 = pd.DataFrame(data=data)
df1.rename(columns={0: feature_names[0], 1: feature_names[1], 2: feature_names[2], 3: feature_names[3],
 4: feature_names[4], 5: feature_names[5], 6: feature_names[6], 7: feature_names[7]}, inplace=True)
df2 = pd.DataFrame(data=target)
df2.rename(columns={0: ‘Target’}, inplace=True)
housing = pd.concat([df1, df2], axis=1)
print(housing.columns)