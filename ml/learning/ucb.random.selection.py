"""
Random Selection test on ads ctr data.
"""
import random
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./../../data/ads.ctr.csv')

N = 10000
d = 10
ads_selected = []
total_reward = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = df.values[n, ad]
    total_reward = total_reward + reward

plt.hist(ads_selected)
plt.title('Histogram of Ad Selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each Ad was Selected')
plt.show()
