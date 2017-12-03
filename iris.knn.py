"""
KNeighbors Classifier test on iris data.
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plot

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
print("X shape / y shape: ", X.shape, y.shape)
print("X train / X test shape: ", X_train.shape, X_test.shape)
print("y train / y test shape: ", y_train.shape, y_test.shape)

k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_predicted = knn.predict(X_test)
    k_scores.append(metrics.accuracy_score(y_test, y_predicted))

cv_range = list(range(1, 31))
cv_scores = []
for cv in cv_range:
    knn = KNeighborsClassifier(n_neighbors=cv)
    cv_scores.append(cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean())

model = KNeighborsClassifier(n_neighbors=16)
model.fit(X_train, y_train)
y_predicted = model.predict(X_test)
print("Highest [knn=16] accuracy score: ", metrics.accuracy_score(y_test, y_predicted))
print("Cross-validation mean accuracy score: ",
      cross_val_score(model, X, y, cv=10, scoring='accuracy').mean())

plot.plot(k_range, k_scores)
plot.xlabel('Value of K for KNN')
plot.ylabel('Testing Accuracy')
plot.show()

plot.plot(cv_range, cv_scores)
plot.xlabel('Value of K for KNN')
plot.ylabel('Cross-Validation Accuracy')
plot.show()
