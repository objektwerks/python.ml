"""
Naive Bayes test on iris data.
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

model = GaussianNB()
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)

print("Accuracy score: ", metrics.accuracy_score(y_test, y_predicted))
