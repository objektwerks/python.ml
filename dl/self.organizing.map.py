"""
Self Organizing Map test on credit card app data.
"""
import numpy as np
import pandas as pd
from pylab import bone, pcolor, colorbar, plot, show
from sklearn.preprocessing import MinMaxScaler
from somlib import MiniSom

df = pd.read_csv('./../data/credit.card.apps.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
print("X shape / y shape: ", X.shape, y.shape)

sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

mappings = som.win_map(X)
frauds = np.concatenate((mappings[(8,1)], mappings[(6,8)]), axis = 0)
frauds = sc.inverse_transform(frauds)
print("Frauds:\n")
print(frauds)
