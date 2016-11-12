import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
import pylab as P

from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv('InputTraumatic.csv')
# print(dataset)


data = dataset.dance.reshape((len(dataset.dance), 1))
CV = dataset.health.reshape((len(dataset.health), 1))

regr = linear_model.LinearRegression()
regr.fit(data,CV)
predicted_results = regr.predict(data)

plt.plot(data, predicted_results, color = 'green', linewidth =3)
plt.scatter(data, CV, color='black')
plt.title('Mental health as a function of song dance level')
plt.xlabel('dance level')
plt.ylabel('health')
plt.show()

print("Result with Outlier:")
print('Coefficients (m): \n', regr.coef_)
print('Intercept (b): \n', regr.intercept_)

# MSE = mean_squared_error(dataset.health, predicted_results)
# RMSE = math.sqrt(MSE)

# R2 = r2_score(dataset.health, predicted_results)

# print("Mean residual sum of squares =", MSE)
# print("RMSE =", RMSE)
# print("R2 =", R2)
# print("Mean residual sum of squares = %.2f" % np.mean((regr.predict(data) - dataset.health) **2))
# print('R2 = %.2f' % regr.score())
