from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)

print(metrics.accuracy_score(y_test, y_predicted))
