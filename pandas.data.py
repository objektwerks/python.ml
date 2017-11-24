"""
Pandas using Chipolte order data.
"""
import pandas as panda

dataframe = panda.read_table('./data/chipolte.tsv')
print("Data shape: ", dataframe.shape)
print("Data:\n", dataframe.head(n=3))
