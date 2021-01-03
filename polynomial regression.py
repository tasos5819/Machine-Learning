import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#linear regrssion model
lin_reg = LinearRegression()
lin_reg.fit(X,y)
#polyomial regression model
#create a martix of power of features and use it in a linear regressor model
poly_reg = PolynomialFeatures(degree = 2 ) #tsekare to degree na deis to fit
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)


#visualization of linear regression
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X),color = 'blue')
plt.title('linear regression model')
plt.xlabel('position')
plt.ylabel('salary')

#visualization of polynomial regression
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.title('linear regression model')
plt.xlabel('position')
plt.ylabel('salary')


