#Polynomial regression

#import libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#building a Linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)
y_pred = lin_reg.predict(x)

#building a polynomial regression 
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)
y_pred_2 = lin_reg_2.predict(poly_reg.fit_transform(x))

#visualizing linear regression
plt.scatter(x,y,color = 'red')
plt.plot(x,y_pred,color = 'blue')
plt.title('Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#visualizing polynomial regression
plt.scatter(x,y,color = 'red')
plt.plot(x,y_pred_2,color = 'blue')
plt.title('Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()

#Predicting salary with linear regression
lin_reg_2.predict(6.5)

#Predictinng salary with polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))














