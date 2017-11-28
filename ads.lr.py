"""
Linear Regression using pandas and advertising data.
"""
import pandas as panda
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sb

dataframe = panda.read_csv('./data/ads.csv', index_col=0)
print("Data shape: ", dataframe.shape)
print("Data:\n", dataframe.head(n=3))

X = dataframe[['TV', 'Radio']]
print("X type: ", type(X))
print("X shape: ", X.shape)
print("X:\n", X.head(n=3))

y = dataframe['Sales']
print("y type: ", type(y))
print("y shape: ", y.shape)
print("y:\n", y.head(n=3))

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print("X train / X test shape: ", X_train.shape, X_test.shape)
print("y train / y test shape: ", y_train.shape, y_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)
print("Root Mean Squared Error (RMSE): ", np.sqrt(metrics.mean_squared_error(y_test, y_predicted)))
print("Cross-validation mean RMSE: ",
      np.sqrt(-cross_val_score(model, X, y, cv=10, scoring='neg_mean_squared_error')).mean())

sb.pairplot(dataframe, x_vars=['TV', 'Radio'], y_vars='Sales', size=7, aspect=0.7, kind='reg')
plot.show()
