"""
Multiple models run against iris data.
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics

X, y = load_iris(return_X_y=True)
print("X shape / y shape: ", X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
print("X train / X test shape: ", X_train.shape, X_test.shape)
print("y train / y test shape: ", y_train.shape, y_test.shape)

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
dtc_predicted = dtc.predict(X_test)
print("Decision Tree Classifier accuracy score:", metrics.accuracy_score(y_test, dtc_predicted))

knn = KNeighborsClassifier(n_neighbors=16)
knn.fit(X_train, y_train)
knn_predicted = knn.predict(X_test)
print("K Neighbors Classifier accuracy score:", metrics.accuracy_score(y_test, knn_predicted))

lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_predicted = lr.predict(X_test)
print("Logistic Regression accuracy score: ", metrics.accuracy_score(y_test, lr_predicted))

gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_predicted = gnb.predict(X_test)
print("Gaussian Naive Bayes accuracy score: ", metrics.accuracy_score(y_test, gnb_predicted))

svm = SVC()
svm.fit(X_train, y_train)
svm_predicted = svm.predict(X_test)
print("Support Vector Machine accuracy score: ", metrics.accuracy_score(y_test, svm_predicted))
