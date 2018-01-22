"""
Pandas features.
"""
import pandas as pd

ads = pd.read_csv('./../data/ads.csv', index_col=0).dropna()
print("Ads shape: ", ads.shape)
print("Ads data:\n", ads.head(n=3))
print("Ad sales >= 20.0:\n", ads[ads.Sales >= 20.0].sort_values('Sales', ascending=False).head(n=3))
print("Ad average sales:\n", ads.mean(axis='index'))

orders = pd.read_table('./../data/orders.tsv', index_col=0).dropna()
orders.drop(['choice_description'], axis=1, inplace=True)
print("Orders shape: ", orders.shape)
print("Orders data:\n", orders.sort_values('item_price', ascending=False).head(n=3))
print("Orders average total:\n", orders.mean(axis='index'))

job_cols = ['id', 'age', 'gender', 'job', 'zip']
jobs = pd.read_table('./../data/jobs.psv', sep='|', header=None, names=job_cols, index_col=0).dropna()
print("Jobs shape: ", jobs.shape)
print("Jobs data:\n", jobs.sort_values('job').head(n=3))
print("Job by age >= 30 and gender = M:\n", jobs[(jobs.age >= 30) & (jobs.gender == 'M')]
      .sort_values('age', ascending=False).head(n=3))
print("Geeks:\n", jobs[jobs.job.isin(['programmer', 'technician'])].head(n=3))
print("Jobs average age:\n", jobs.mean(axis='index'))
print("Jobs gender/job crosstab:\n", pd.crosstab(jobs.gender, jobs.job))

ufos = pd.read_csv('./../data/ufos.csv').dropna()
ufos['Location'] = ufos.City + ', ' + ufos.State
ufos.rename(columns={'Time': 'Date_Time'}, inplace=True)
ufos.drop(['Color', 'State'], axis=1, inplace=True)
print("UFOs shape: ", ufos.shape)
print("UFOs data:\n", ufos.sort_values('Location').head(n=3))

movies = pd.read_csv('./../data/imdb.csv').dropna()
print("Movies shape: ", movies.shape)
print("Movie ratings:\n", movies.sort_values('stars', ascending=False).head(n=3))
print("Movie genres:\n", movies.genre.describe())

drinks = pd.read_csv('./../data/drinks.csv').dropna()
print("Drinks shape: ", drinks.shape)
print("Drinks litres:\n", drinks.sort_values('litres', ascending=False).head(n=3))
print("Drinks describe:\n", drinks.describe())
print("Beer by continent:\n", drinks.groupby('continent').beer.agg(['mean', 'max']))
