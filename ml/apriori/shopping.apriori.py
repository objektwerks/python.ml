"""
Apriori test on shopping data.
"""
import pandas as pd
from apyori import apriori

df = pd.read_csv('./../../data/shopping.csv', header = None)
baskets = []
for i in range(0, 7501):
    baskets.append([str(df.values[i, j]) for j in range(0, 20)])
records = apriori(baskets, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
j = 0
for record in records:
    print(record)
    j += 1
    if j == 10:
        break
