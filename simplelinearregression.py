from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
#datapreprocessing
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]
#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                               random_state=0)
#make the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#perdicting the test set results
y_pred = regressor.predict(X_test)
#visualising training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title('salary vs experience (training set)')
plt.xlabel('years of experience')
plt.ylabel('salary')

#visualising test set results
plt.scatter(X_test,y_test,color='red')
plt.plot(X_train,regressor.predict(X_train), color='blue')
plt.title('salary vs experience (test set)')
plt.xlabel('years of experience')
plt.ylabel('salary')

#find the coefficients of linear regression
#y=b1+box
#b0
print(regressor.coef_)
#b1
print(regressor.intercept_)

#make a simple prediction of the salary for an employ that works 12 years
print(regressor.predict([[12]]))





