"""
Support Vector Regression test on roles.salaries data.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./../../data/roles.salaries.csv')
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values
print("X shape / y shape: ", X.shape, y.shape)
print("X:\n", X)
print("y:\n", y)

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = y.reshape(-1, 1)
y = sc_y.fit_transform(y)
print("Feature scaled X:\n", X)
print("Feature scaled y:\n", y)

model = SVR(kernel = 'rbf')
model.fit(X, y)

y_predicted = sc_y.inverse_transform(model.predict(sc_X.transform(np.array([[6.5]]))))
print("Support Vector Regression predict @ 6.5 role-level salary: ", y_predicted)

plt.scatter(X, y, color = 'red')
plt.plot(X, model.predict(X), color = 'blue')
plt.title('Support Vector Regression')
plt.xlabel('Role-Level')
plt.ylabel('Salary')
plt.show()

# Role-Level 1 - 10, step 0.1
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')
plt.plot(X_grid, model.predict(X_grid), color = 'blue')
plt.title('Smooth Support Vector Regression')
plt.xlabel('Role-Level')
plt.ylabel('Salary')
plt.show()
