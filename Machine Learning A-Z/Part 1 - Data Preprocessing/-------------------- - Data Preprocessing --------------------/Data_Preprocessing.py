# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 18:09:49 2019

@author: Gouasmia Zakaria 
"""


#Importing the libraries 
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 



#Importing the dataset .
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


#taking care of missing data

from sklearn.preprocessing import Imputer 

imputer = Imputer(missing_values='NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding categprocal data 
#list_of_country = y
#i=0
#for item in list_of_country:
#    if item == 'Yes':
#        list_of_country[i] = 1
#        i=i+1
#    else : 
#          list_of_country[i]= 0 
#          i=i+1
#    

from sklearn.preprocessing import LabelEncoder 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


#dummy Encoding 
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

#Splitting the dataset into training set  and Testing set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size = 0.2 , random_state = 0 )

#Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

