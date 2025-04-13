#final project for Data Analytics course
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#import data
filepath='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df = pd.read_csv(filepath, header=0)

#clean up data
df.drop(['id','Unnamed: 0'], axis=1, inplace=True)
mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)
mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)

#explore data
floor_values = df['floors'].value_counts()
floordf = floor_values.to_frame()
print(floordf, type(floordf))

#boxplot to see if many outliers for any category on a variable, use waterfront variable
sns.boxplot (x="waterfront", y="price",data=df)
plt.ylim(0,)

#regplot it to see if correlation
sns.regplot (x="sqft_above", y="price",data=df)
plt.ylim(0,)

#We can use the Pandas method corr() to find the feature other than price that is most correlated with price.
#first build df with just the data types of numbers
df_numeric = df.select_dtypes(include=[np.number])
#corr score them and sort for easy viewing
df_numeric.corr()['price'].sort_values()

X = df[['sqft_living']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]  
X1 = df[features]
lm.fit(X1,Y)
lm.score(X1, Y)

Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe=Pipeline(Input)
X1 = X1.astype(float)
pipe.fit(X1,Y)
ypipe=pipe.predict(X1)
print(r2_score(Y,ypipe))

#evaluate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

from sklearn.linear_model import Ridge
RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(x_train, y_train)
yhat = RidgeModel.predict(x_test)
print(r2_score(y_test,yhat))

pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.transform(x_test)
RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(x_train_pr, y_train)
y_hat = RidgeModel.predict(x_test_pr)
print(r2_score(y_test,y_hat))

df.head()
df.info()
df.describe()