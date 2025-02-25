Python ML
---------
Python 3 machine and deep learning algo scripts using Scikit-learn, Pandas, Numpy, Matplotlib, Xgboost,
Keras, Theano, TensorFlow, PyTorch, PySpark, Seaborn and Statsmodels.

Todo
----
1. After 6+ years of change in the Python / ML space, each script needs to be revisited.

Virtual Env
-----------
1. python3.13 -m venv venv
2. source venv/bin/activate
3. pip3 list
4. pip3 install --upgrade pip ( optional )
5. pip3 freeze > requirements.txt ( optional )
>See [VE Setup](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/)

Install
-------
1. pip3 install scikit-learn, pandas, matplotlib, xgboost, keras, tensorflow, torch, torchvision, torchaudio, nltk, seaborn, statsmodels
2. pip3 install theano ( may fail )
3. pip3 install pyspark
4. pip3 list
5. pip3 freeze > requirements.txt
>**Note:** All modules can be installed via: ```pip3 install -r requirements.txt```

Run
---
1. cd src/**/*.py
2. python3.13 *.py

PyCharm Run
-----------
1. right-click *.py > run

Notes
-----
1. Jupiter notebook ( ```src/nb/notebook.ipynb``` ) requires PyCharm Professional.

Courses
-------
ML and DL scripts are based on these **Udemy** courses:
  1. Machine Learning A-Z: Hands-on Python & R in Data Science
  2. Deep Learning A-Z: Hands-on Artificial Neural Networks

Data
----
All data is located in the data directory, less 1 dataset for ```src/dl/convolution.neural.network.py```,
which contains instructions.

Machine Learning
----------------
The following ML algos are tested:

**Regression:** Linear, Logistic, Polynomial, Decision Tree, Random Forrest, Support Vector

**Classification:** Naive Bayes, Linear Support Vector Machine, Gaussian RBF Support Vector Machine,
K-Nearest Neighbors, Decision Tree, Random Forrest, Logistic Regression, K-Fold Cross Validation,
Grid Search, XGBoost

**Clustering:** K-Means, Hierarchical

**Learning:** Apriori, Eclat, Upper Confidence Bound (UCB), Thompson Sampling, Natural Language Processing (NLP)

**Reduction:** Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), Kernel Principal Component Analysis (PCA)

**Libraries:** Scikit-learn, Pandas, Numpy, Matplotlib, Xgboost

Deep Learning
-------------
The following DL algos are tested:

**Neural Networks:** Artificial, Convolution, Self Organizing Map, Boltzmann Machine, Autoencoder, Recurrent

**Libraries:** Keras, Theano, TensorFlow, PyTorch
