# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:36:52 2020

@author: kingslayer
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv(r"Salary_Data.csv")

#creating matrix of features
X=dataset.iloc[:,0:1].values

#creating dependant vector 
y=dataset.iloc[:,1].values

#splitting into training and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#Creating and fitting the linear model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)

#predicting results
y_pred=regressor.predict(X_test)

#plotting the graph for training set and test set
plt.scatter(X_train,y_train,color="green")
plt.scatter(X_test,y_test,color="red")
plt.plot(X_train,regressor.predict(X_train),color="blue")
plt.title("SALARY VS EXPERIENCE")
plt.xlabel("years of experience")
plt.ylabel("Salary")
plt.show()

#predicting a salary
y_pred1=regressor.predict([[6]])