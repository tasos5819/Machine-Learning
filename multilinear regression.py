import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])],
                       remainder='passthrough')
X=np.array(ct.fit_transform(X))


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2,
                                                    random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)


#model evaluation
y_pred = regressor.predict(X_test)
vector = np.concatenate((y_pred.reshape(len(y_pred),1),
                      y_test.reshape(len(y_test),1)),axis=1)


#one single predict: state=califorina, marketing spend = 
#300000, rd spend= 16000, admin spend

print(regressor.predict([[1,0,0,16000,130000,300000]]))

#find the coefficients
#b_i
print(regressor.coef_)
#b_0
print(regressor.intercept_)
