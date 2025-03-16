import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

filepath='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(filepath, header=0)
'''
print(df.head())

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
