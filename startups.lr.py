"""
Linear Regression test on startups data.
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm

df = pd.read_csv('./data/startups.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 4].values

state_encoder = LabelEncoder()
X[:, 3] = state_encoder.fit_transform(X[:, 3])

state_hot_encoder = OneHotEncoder(categorical_features = [3])
X = state_hot_encoder.fit_transform(X).toarray()

# Remove [dummy vars - 1] to avoid dummy var trap.
# Done automatically by sklearn linear regression.
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print("X shape / y shape: ", X.shape, y.shape)
print("X train / X test shape: ", X_train.shape, X_test.shape)
print("y train / y test shape: ", y_train.shape, y_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)

plt.scatter(y_test, y_predicted, color = 'red')
plt.title('Predicted Profit vs Profit')
plt.xlabel('Profit')
plt.ylabel('Predicited Profit')
plt.show()

# Prepend x0 to X.
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

# Backward elimination.
model_OLS = sm.OLS(endog = y, exog = X[:, [0, 1, 2, 3, 4, 5]]).fit()
print("OLS Summary [0, 1, 2, 3, 4, 5]\n", model_OLS.summary())

model_OLS = sm.OLS(endog = y, exog = X[:, [0, 1, 3, 4, 5]]).fit()
print("OLS Summary [0, 1, 3, 4, 5]\n", model_OLS.summary())

model_OLS = sm.OLS(endog = y, exog = X[:, [0, 3, 4, 5]]).fit()
print("OLS Summary [0, 3, 4, 5]\n", model_OLS.summary())

model_OLS = sm.OLS(endog = y, exog = X[:, [0, 3, 5]]).fit()
print("OLS Summary [0, 3, 5]\n", model_OLS.summary())

# R&D Spend is the best independent variable.
model_OLS = sm.OLS(endog = y, exog = X[:, [0, 3]]).fit()
print("OLS Summary [0, 3]\n", model_OLS.summary())
