"""
Logistic Regression test on purchases data.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import Imputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler

df = pd.read_csv('./../../../data/purchases.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 3].values
print("X: ", X)
print("y: ", y)

age_salary_imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
age_salary_imputer = age_salary_imputer.fit(X[:, 1:3])
X[:, 1:3] = age_salary_imputer.transform(X[:, 1:3])
print("X age-salary nan-to-mean imputer:\n", X)

country_encoder = LabelEncoder()
X[:, 0] = country_encoder.fit_transform(X[:, 0])
print("X country label encoder:\n", X)

country_hot_encoder = OneHotEncoder()
X = country_hot_encoder.fit_transform(X).toarray()
print("X country hot encoder:\n", X)

purchased_encoder = LabelEncoder()
y = purchased_encoder.fit_transform(y)
print("y purchased label encoder: ", y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print("X/y train-test split!")

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

print("X shape / y shape: ", X.shape, y.shape)
print("X train / X test shape: ", X_train.shape, X_test.shape)
print("y train / y test shape: ", y_train.shape, y_test.shape)

model = LogisticRegression()
model.fit(X_train, y_train)
print("y predicted: ", model.predict(X_test))
