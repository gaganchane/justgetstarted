import pandas as pd
import numpy as np
from sklearn import linear_model

data = pd.read_csv('../data_all.csv')

dataset = data[data['traumatic_experience'] == 'Yes']

x = dataset[['tempo', 'popularity','energy','liveness','dance','valence','instrumental','acoustic']]

y = dataset['health']

print(y)

clf = linear_model.LinearRegression()

clf.fit(x,y)

print(clf.coef_)
print(clf.get_params())
print(clf.score(x,y))