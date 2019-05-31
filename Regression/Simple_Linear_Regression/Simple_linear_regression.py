#Simple Linear Regression

#Predicting salary of an employee based on years of experience

#importing libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#splitting dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state=1) 

#fitting the model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting the result
y_predict = regressor.predict(x_test)

#visualize the training dataset
plt.scatter(x_train,y_train,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('Salary vs Experience [Training dataset]')
plt.xlabel('Number of Experience in years')
plt.ylabel('Salary')
plt.show()

#visualize the test dataset
plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color = 'blue')
plt.title('Salary vs Experience [Testing dataset]')
plt.xlabel('Number of Experience in years')
plt.ylabel('Salary')
plt.show()

