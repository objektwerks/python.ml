"""
Decision Tree Regression test on roles.salaries data.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('./../../../data/roles.salaries.csv')
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values
print("X shape / y shape: ", X.shape, y.shape)
print("X:\n", X)
print("y:\n", y)

model = DecisionTreeRegressor(random_state=0)
model.fit(X, y)
print("Decision Tree Regression predict @ 6.5 role-level salary: ", model.predict(6.5))

# Role-Level 1 - 10, step 0.1
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color='red')
plt.plot(X_grid, model.predict(X_grid), color='blue')
plt.title('Decision Tree Regression')
plt.xlabel('Role-Level')
plt.ylabel('Salary')
plt.show()
