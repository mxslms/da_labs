import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

filepath='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(filepath, header=0)

print(df.head())
'''
sdfsdfsdfsdfsd

print(df['peak-rpm'].dtypes) 

numericdf = df.select_dtypes(include=['float64','int64'])
print(numericdf.head())
corr_matrix = numericdf.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr_matrix, annot=True), cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
corr_matrix
df[['bore', 'stroke', 'compression-ratio', 'horsepower']].corr()    

sns.regplot(x='engine-size', y='price', data=df)
plt.ylim(0,)
plt.show()

df[["engine-size", "price"]].corr()
df[["highway-mpg", "price"]].corr()
sns.regplot(x='highway-mpg', y='price', data=df)
plt.ylim(0,)
plt.show()

sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)
plt.show()
df[['peak-rpm','price']].corr()

sns.regplot(x="stroke", y="price", data=df)
plt.ylim(0,)
plt.show()
df[['stroke','price']].corr()

sns.boxplot(x="body-style", y="price", data=df)
plt.show()

sns.boxplot(x="engine-location", y="price", data=df)
plt.show()

sns.boxplot(x="drive-wheels", y="price", data=df)
plt.show()
'''
df.describe(include=['object'])
df.describe()
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.reset_index(inplace=True)
drive_wheels_counts=drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'})
drive_wheels_counts


engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)

df['drive-wheels'].unique()

df_group_one = df[['drive-wheels','body-style','price']]

df_grouped = df_group_one.groupby(['drive-wheels'], as_index=False).agg({'price': 'mean'})
df_grouped

df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_test1

grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot

grouped_pivot = grouped_pivot.fillna(0) #fill missing values with 0
grouped_pivot

sns.heatmap(grouped_pivot, annot=True, cmap='YlGnBu')  # annot=True shows values in cells
plt.show()

df_grpstyle = df[['body-style','price']]
df_grpstyle = df_grpstyle.groupby(['body-style'], as_index=False).mean()
df_grpstyle 

plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()

df.select_dtypes(include=['number']).corr()
from scipy import stats
pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  

pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  

