"""
Pandas to read features.
"""
import pandas as panda

orders = panda.read_table('./data/chipolte.tsv', index_col=0)
print("Orders shape: ", orders.shape)
print("Orders data:\n", orders.head(n=3))

occupation_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
occupations = panda.read_table('./data/occupations.psv', sep='|', header=None, names=occupation_cols, index_col=0)
print("Occupations shape: ", occupations.shape)
print("Occupations data:\n", occupations.head(n=3))
