
from asyncio import CancelledError
from statistics import linear_regression
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import sklearn as sk

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

#(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module']) 

diabetes = datasets.load_diabetes()

diabetes_x = diabetes.data

train_x = diabetes_x[:-30]
test_x = diabetes_x[-30:]

train_y = diabetes.target[:-30]
test_y = diabetes.target[-30:]


model = linear_model.LinearRegression()

model.fit(train_x, train_y)

diabetes_predicted = model.predict(test_x)

print("Mean Squared Error is:", mean_squared_error(test_x, test_y))

print("Weights:", model.coef_)
print("Intercept:", model.intercept_)

# plt.scatter(test_x, test_y, color="blue")
# plt.plot(test_x, diabetes_predicted)

# plt.show()

# Mean Squared Error is: 22916.762209847675
# Weights: [941.43097333]
# Intercept: 153.39713623331644








