#Support vector Regression

#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset 
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train = sc_x.fit_transform(x)
y_train = sc_y.fit_transform(y)

"""
#encoding column
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,3] = labelencoder_x.fit_transform(x[:,3]) 
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#avoiding dummy variable trap
x = x[:,1:] 

#splitting dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0) 
"""

#Building  Regression
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)

#Predicting with Polynomial regression
#y_pred = regressor.predict(6.5)

#Visualizing regression
plt.scatter(x,y,color = 'red')
plt.plot(x,regressor.predict(x),color = 'blue')
plt.title('Salary vs Years of experience[SVM regression]')
plt.ylabel('Salary')
plt.xlabel('Years of experience')
plt.show()








