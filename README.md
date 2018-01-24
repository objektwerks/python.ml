Python ML
---------
The principal purpose of this project is to build and test machine and deep learning algos.

Python
------
I used Python 3 throughout this project.

Libraries
---------
I initially went down the Anaconda path. But it created issues with Homebrew, my one and only
package manager. So I transitioned to Pip3 to install all required libraries. The installation
of xgboost was challenging --- see churn.modeling.xgboost.py for details.

Sources
-------
Most of the material contained herein is based on 2 Udemy online courses:
  1. Machine Learning A-Z: Hands-on Python & R in Data Science
  2. Deep Learning A-Z: Hands-on Artificial Neural Networks ( a work-in-progress )

Data
----
All required data is located in the data directory, less 1 dataset for cats.dogs.cnn.py, which
contains instructions.

Machine Learning
----------------
The following ML algos are tested:

**Regression:** Linear, Logisitic, Polynominal, Decision Tree, Random Forrest, Support Vector

**Classification:** Naive Bayes, Linear Support Vector Machine, Gaussian RBF Support Vector Machine,
K-Nearest Neighbors, Decision Tree, Random Forrest, Logistic Regression, K-Fold Cross Validation,
Grid Search, XGBoost

**Clustering:** K-Means, Hierarchical

**Learning:** Apriori, Eclat, Upper Confidence Bound (UCB), Thompson Sampling, Natural Language Processing (NLP)

**Reduction:** Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), Kernel Principal Component Analysis (PCA)

Deep Learning
-------------
The following DL algos are tested:

**Neural Networks:** Artificial, Convolution