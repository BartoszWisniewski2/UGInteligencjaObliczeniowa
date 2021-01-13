import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

dataForPlot = []
df = pd.read_csv('breast-cancer-wisconsin-removed.csv')

df.drop(df.columns[0], axis=1,inplace=True)
df2 = (df
       .select_dtypes(np.number)
       .agg(['min', 'max', 'mean']))
print(df2)
df2.to_csv('dane.csv', index=False)
for c in df.columns:
       df[c].value_counts(sort=False).plot(kind='bar')
       plt.title(c)
       plt.show()

