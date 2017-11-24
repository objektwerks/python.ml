"""
Pandas read features.
"""
import pandas as panda

ads = panda.read_csv('./data/ads.csv', index_col=0)
print("Ads shape: ", ads.shape)
print("Ads data:\n", ads.head(n=3))

ufos = panda.read_csv('./data/ufos.csv')
assert ufos.City.size > 0
ufos['Location'] = ufos.City + ', ' + ufos.State
assert ufos.Location.size > 0
print("UFOs shape: ", ufos.shape)
print("UFOs data:\n", ufos.head(n=3))

orders = panda.read_table('./data/orders.tsv', index_col=0)
print("Orders shape: ", orders.shape)
print("Orders data:\n", orders.head(n=3))

occupation_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
occupations = panda.read_table('./data/occupations.psv', sep='|', header=None, names=occupation_cols, index_col=0)
print("Occupations shape: ", occupations.shape)
print("Occupations data:\n", occupations.head(n=3))
