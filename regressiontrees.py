import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#values για να φτιαξει πινακα numpy
# δεν χρειαζεται feature scaling σε decission tree regression

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

prediction = regressor.predict([[6.5]])

#visualization
X_grid = np.arange(min(X), max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color= 'blue')
plt.title('truth of bluff (Decision tree regression)')
plt.xlabel('position level')
plt.ylabel('salary')

#δεν δουλευει καλα σε 2d μοντέλα. θελει περισσοτερες διαστασεις