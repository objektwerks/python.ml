from sklearn import datasets
from sklearn import metrics
from sklearn.svm import SVC

dataset = datasets.load_iris()

model = SVC()
model.fit(dataset.data, dataset.target)
print(model)

expected = dataset.target
predicted = model.predict(dataset.data)

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))