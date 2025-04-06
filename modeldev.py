#practice model development with laptop prices
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 

#This function will download the dataset into your browser 

 

lm = LinearRegression()
X = df[['CPU_frequency']]
Y = df[['Price']]
lm.fit(X,Y)
Yhat = lm.predict(X)

ax1 = sns.kdeplot(Y, color='green', label = "acutal value")
sns.kdeplot(Yhat, color='b', label="Fitted values", ax=ax1)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of laptops')
plt.legend(['Actual Value', 'Predicted Value'])
plt.show()

mse_lsr = mean_squared_error(Y,Yhat)
r2score_lsr = r2_score(Y,Yhat)
print (mse_lsr)
print (r2score_lsr)

sns.regplot(x=X, y=Y)
plt.ylim(0,)
plt.show()

print (X)
print (Yhat)

plt.scatter(X,Y)
plt.show()

'''poly1model = np.poly1d(np.polyfit(X['CPU_frequency'],Y['Price'], 1))
myline = np.linspace(min(X), max(X))
plt.scatter(X,Y)
plt.plot(myline, poly1model(myline))
plt.show()'''
X = X['CPU_frequency'].to_numpy().flatten()
f1 = np.polyfit(X,Y['Price'],1)
p1 = np.poly1d(f1)
r2score_poly1 = r2_score(Y['Price'],p1(X))
f3 = np.polyfit(X,Y['Price'],3)
p3 = np.poly1d(f3)
r2score_poly3 = r2_score(Y['Price'],p3(X))
f5 = np.polyfit(X,Y['Price'],5)
p5 = np.poly1d(f5)
r2score_poly5 = r2_score(Y['Price'],p5(X))

print(r2score_poly1, r2score_poly3, r2score_poly5)



Z = df[['CPU_frequency','RAM_GB','Storage_GB_SSD','CPU_core','OS','GPU','Category']]
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
Z = Z.astype(float)
pipe.fit(Z,Y)
ypipe=pipe.predict(Z)
print(ypipe)
print('MSE for multi-variable polynomial pipeline is: ', mean_squared_error(Y['Price'], ypipe))
print('R^2 for multi-variable polynomial pipeline is: ', r2_score(Y['Price'], ypipe))

ax1 = sns.kdeplot(Y['Price'], color='green', label = "acutal value")
sns.kdeplot(ypipe, color='b', label="Fitted values", ax=ax1)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of laptops')
plt.legend(['Actual Value', 'Predicted Value'])
plt.show()

plt.scatter(Y['Price'], ypipe, color='blue', label='Data Points')
plt.scatter(Y['Price'], ypipe, color='blue', label='Data Points')