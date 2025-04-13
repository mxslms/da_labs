import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

#import data
filepath='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'
df = pd.read_csv(filepath, header=None)

#clean up data - label, swap bad with nan, replace nans with avgs or category mode, convert types
df.columns = ['age','gender','bmi','no_of_children','smoker','region','charges']
df.replace('?', np.nan, inplace=True)

avg_age=df['age'].astype('float').mean(axis=0)
df.replace({'age':np.nan},avg_age, inplace=True)

smoker_mode = df['smoker'].mode()[0]
df.replace({'smoker':np.nan},smoker_mode, inplace=True)

df[['age','smoker']]=df[['age','smoker']].astype(int)

df['charges'] = df['charges'].round(2)

#explore
sns.regplot (x="bmi", y="charges",data=df)
plt.ylim(0,)

sns.boxplot (x="smoker", y="charges",data=df)
plt.ylim(0,)

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

#build a model
#smoker model
lm=LinearRegression()
X=df[['smoker']]
Y=df['charges']
lm.fit(X,Y)
print(lm.score(X, Y))

#other attribute model
lm1=LinearRegression()
X1=df.drop(['charges'], axis=1)
lm1.fit(X1,Y)
print(lm1.score(X1, Y))

#use a pipeline to preprocess by scaling and adding poly features then model with LR
Input=[('scale',StandardScaler()),('polynomial',PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe=Pipeline(Input)
X1 = X1.astype(float)
pipe.fit(X1,Y)
ypipe=pipe.predict(X1)
print(r2_score(Y,ypipe))

#split data to train and test
x_train, x_test, y_train, y_test = train_test_split(X1, Y, test_size=0.2, random_state = 1)
RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(x_train, y_train)
yhat = RidgeModel.predict(x_test)
print(r2_score(y_test,yhat))

# x_train, x_test, y_train, y_test hold same values as in previous cells
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.transform(x_test)
RidgeModel.fit(x_train_pr, y_train)
y_hat = RidgeModel.predict(x_test_pr)
print(r2_score(y_test,y_hat))





print(df.head())
df.info()
df.describe()
print(type(x_train))
print(x_test_pr)
