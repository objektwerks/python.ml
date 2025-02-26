Python ML
---------
>Python 3 machine and deep learning scripts using Scikit-learn, Pandas, Numpy, Matplotlib, Xgboost,
>Keras, Theano, TensorFlow, PyTorch, PySpark, Seaborn and Statsmodels.

News
----
1. [2025.2.25] Converted this pip project to uv.

Todo
----
1. After 6+ years of change in the Python / ML space, each script needs to be revisited.

Repository
----------
* [PyPi](https://pypi.org/)

Development
-----------
>To install a Python development environment:
1. Install [Homebrew](https://brew.sh/)
2. brew install python@3.13
3. brew install mypy
4. Install [VSCode](https://code.visualstudio.com/)
5. Install VSCode Python Microsoft Extensions: Python, Python Debugger, Pylance

Virtual Env
-----------
1. python3.13 -m venv venv
2. source venv/bin/activate
3. pip3 install --upgrade pip
4. pip3 install scikit-learn, pandas, matplotlib, xgboost, keras, torch, torchvision, torchaudio, nltk, seaborn, statsmodels
5. pip3 install tensorflow ( may fail )
6. pip3 install theano ( may fail )
7. pip3 install pyspark
8. pip3 list
9. pip3 freeze > requirements.txt
>**Torch** and **Tensorflow** only support Python 3.12. See [VE Setup](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/)

Convert to UV
-------------
>To install:
1. brew install pipx
2. pipx install uv ( a global ```brew install uv``` works, yet appears to fail at the project level )
>To convert:
1. uv init
2. uv add -r requirements.txt
3. uv sync

Install Dependency
------------------
>To install a dependency:
1. pip3 install dependency-1
2. pip3 freeze > requirements.txt
>or:
1. uv add 'dependency-1'

Install Dependencies
--------------------
>To install dependencies:
1. pip3 install dependency-1, dependency-2
>or:
1. uv add -r 'dependency-1' 'dependency-2'

Upgrade Dependencies
--------------------
>To upgrade dependencies in **requirements.txt**:
1. pip3 install --upgrade -r requirements.txt
>or:
1. uv sync

Run
---
>To run a script, replace *.py with a py script file name:
1. python3.13 ./src/**/*.py
>or:
1. uv run ./src/**/*.py

Notes
-----
1. Jupiter notebook ( ```src/nb/notebook.ipynb``` ) requires PyCharm Professional.

Courses
-------
>ML and DL scripts are based on these **Udemy** courses:
1. Machine Learning A-Z: Hands-on Python & R in Data Science
2. Deep Learning A-Z: Hands-on Artificial Neural Networks

Data
----
>All data is located in the data directory, less 1 dataset for ```src/dl/convolution.neural.network.py```, which contains instructions.

Machine Learning
----------------
>The following ML algos are tested:

>**Regression:** Linear, Logistic, Polynomial, Decision Tree, Random Forrest, Support Vector

>**Classification:** Naive Bayes, Linear Support Vector Machine, Gaussian RBF Support Vector Machine,
K-Nearest Neighbors, Decision Tree, Random Forrest, Logistic Regression, K-Fold Cross Validation,
Grid Search, XGBoost

>**Clustering:** K-Means, Hierarchical

>**Learning:** Apriori, Eclat, Upper Confidence Bound (UCB), Thompson Sampling, Natural Language Processing (NLP)

>**Reduction:** Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), Kernel Principal Component Analysis (PCA)

>**Libraries:** Scikit-learn, Pandas, Numpy, Matplotlib, Xgboost

Deep Learning
-------------
>The following DL algos are tested:

>**Neural Networks:** Artificial, Convolution, Self Organizing Map, Boltzmann Machine, Autoencoder, Recurrent

>**Libraries:** Keras, Theano, TensorFlow, PyTorch