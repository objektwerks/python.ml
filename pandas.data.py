"""
Pandas read features.
"""
import pandas as panda

ads = panda.read_csv('./data/ads.csv', index_col=0)
print("Ads shape: ", ads.shape)
print("Ads data:\n", ads.head(n=3))
print("Ad sales >= 20.0:\n", ads[ads.Sales >= 20.0].sort_values('Sales', ascending=False).head(n=3))
print("Ad average sales:\n", ads.mean(axis='index'))

orders = panda.read_table('./data/orders.tsv', index_col=0)
orders.drop(['choice_description'], axis=1, inplace=True)
print("Orders shape: ", orders.shape)
print("Orders data:\n", orders.sort_values('item_price', ascending=False).head(n=3))
print("Orders average total:\n", orders.mean(axis='index'))

occupation_cols = ['id', 'age', 'gender', 'occupation', 'zip']
occupations = panda.read_table('./data/occupations.psv', sep='|', header=None, names=occupation_cols, index_col=0)
print("Occupations shape: ", occupations.shape)
print("Occupations data:\n", occupations.sort_values('occupation').head(n=3))
print("Occupation by age >= 30 and gender = M:\n", occupations[(occupations.age >= 30) \
    & (occupations.gender == 'M')].sort_values('occupation').head(n=3))
print("Geeks:\n", occupations[occupations.occupation.isin(['programmer', 'technician'])].head(n=3))
print("Occupations average age:\n", occupations.mean(axis='index'))

ufos = panda.read_csv('./data/ufos.csv')
assert ufos.City.size > 0
ufos['Location'] = ufos.City + ', ' + ufos.State
assert ufos.Location.size > 0
ufos.rename(columns={'Time': 'Date_Time'}, inplace=True)
assert ufos.Date_Time.size > 0
ufos.drop(['Color', 'State'], axis=1, inplace=True)
ufos.drop([0, 1], axis=0, inplace=True)
print("UFOs shape: ", ufos.shape)
print("UFOs data:\n", ufos.sort_values('Location').head(n=3))
