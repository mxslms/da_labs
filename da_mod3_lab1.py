import pandas as pd
import numpy as np
import seaborn as sns

filepath='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv'
df = pd.read_csv(filepath, header=0)

print(df.head())

import matplotlib.pyplot as plt

print(df['peak-rpm'].dtypes) 

numericdf = df.select_dtypes(include=['float64','int64'])
print(numericdf.head())
corr_matrix = numericdf.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr_matrix, annot=True), cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()