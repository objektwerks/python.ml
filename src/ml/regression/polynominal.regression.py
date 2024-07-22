"""
Polynominal Regression test on roles.salaries data.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('./../../data/roles.salaries.csv')
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values
print("X shape / y shape: ", X.shape, y.shape)
print("X:\n", X)
print("y:\n", y)

model = LinearRegression()
model.fit(X, y)

features = PolynomialFeatures(degree=4)
X_poly = features.fit_transform(X)
features.fit(X_poly, y)

ploy_model = LinearRegression()
ploy_model.fit(X_poly, y)

plt.scatter(X, y, color='red')
plt.plot(X, model.predict(X), color='blue')
plt.title('Linear Regression')
plt.xlabel('Role-Level')
plt.ylabel('Salary')
plt.show()

plt.scatter(X, y, color='red')
plt.plot(X, ploy_model.predict(X_poly), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Role-Level')
plt.ylabel('Salary')
plt.show()

# Role-Level 1 - 10, step 0.1
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
X_smooth_poly = features.fit_transform(X_grid)

plt.scatter(X, y, color='red')
plt.plot(X_grid, ploy_model.predict(X_smooth_poly), color='blue')
plt.title('Smooth Polynomial Regression')
plt.xlabel('Role-Level')
plt.ylabel('Salary')
plt.show()

# Polynominal Regression is more accurate!
print("Linear Regression predict @ 6.5 role-level salary: ", model.predict(6.5))
print("Polynominal Regression predict @ 6.5 role-level salary: ", ploy_model.predict(features.fit_transform(6.5)))
