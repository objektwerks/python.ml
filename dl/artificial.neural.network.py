"""
Artificial Neural Network test on churn modeling data.
WARNING: For Python 3.6 install Tensorflow 1.5+ Note the version number in the url!
INSTALL:
1. pip3 instal theano
2. pip3 install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.5.0rc1-py3-none-any.whl
3. pip3 install keras
"""
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./../data/churn.modeling.csv')
X = df.iloc[:, 3:13].values
y = df.iloc[:, 13].values

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print("X shape / y shape: ", X.shape, y.shape)
print("X train / X test shape: ", X_train.shape, X_test.shape)
print("y train / y test shape: ", y_train.shape, y_test.shape)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = Sequential()

# Input Layer
classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))
# Hidden Layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))
# Output Layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(X_train, y_train, batch_size = 10, epochs = 10)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", cm)

"""
WARNING: The KerasClassifier hangs! And the GridSearchCV requires hours to run, still resulting
in only 86% accuracy.
"""