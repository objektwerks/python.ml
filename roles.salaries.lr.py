"""
Linear Polynominal Regression test on roles.salaries data.
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./data/roles.salaries.csv')
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values
print("X shape / y shape: ", X.shape, y.shape)
print("X:\n", X)
print("y:\n", y)

model = LinearRegression()
model.fit(X, y)

features = PolynomialFeatures(degree = 4)
X_poly = features.fit_transform(X)
features.fit(X_poly, y)
print("X poly:\n", X_poly)

ploy_model = LinearRegression()
ploy_model.fit(X_poly, y)

plt.scatter(X, y, color = 'red')
plt.plot(X, model.predict(X), color = 'blue')
plt.title('Linear Regression')
plt.xlabel('Role-Level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X, y, color = 'red')
plt.plot(X, ploy_model.predict(features.fit_transform(X)), color = 'blue')
plt.title('Linear Polynomial Regression')
plt.xlabel('Role-Level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, ploy_model.predict(features.fit_transform(X_grid)), color = 'blue')
plt.title('Smooth Linear Polynomial Regression')
plt.xlabel('Role-Level')
plt.ylabel('Salary')
plt.show()

print("Linear Regression predict: ", model.predict(6.5))
print("Linear Polynominal Regression predict: ", ploy_model.predict(features.fit_transform(6.5)))
