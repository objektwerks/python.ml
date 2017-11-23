"""
Linear Regression using pandas against advertising data.
"""
import pandas as panda

data = panda.read_csv('ad.csv', index_col=0)
print(data.shape)
print(data.head(n=3))
