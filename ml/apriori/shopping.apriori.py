"""
Apriori test on shopping data.
"""
import pandas as pd
from apyori import apriori

df = pd.read_csv('./../../data/shopping.csv', header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(df.values[i, j]) for j in range(0, 20)])
records = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
print("Results:\n", list(records))
