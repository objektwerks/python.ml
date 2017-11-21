"""
KNeighbors Classifier test on iris data.
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plot

X, y = load_iris(return_X_y=True)
print("X shape: ", X.shape)
print("y shape: ", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
print("X train shape: ", X_train.shape)
print("X test shape: ", X_test.shape)
print("y train shape: ", y_train.shape)
print("y test shape: ", y_test.shape)

k_range = list(range(1, 26))
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_predicted = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_predicted))

model = KNeighborsClassifier(n_neighbors=16)
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)
print("Highest [knn = 16] accuracy score: ", metrics.accuracy_score(y_test, y_predicted))

plot.plot(k_range, scores)
plot.xlabel('Value of K for KNN')
plot.ylabel('Testing Accuracy')
plot.show()
