# -*- coding: utf-8 -*-
"""
Created on Fri May 25 16:50:52 2018

@author: Sybille Legitime
"""

#Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Avoiding the Dummy Variable Trap
X = X[:, 1:] #taking one dummy variable away manually to avoid redundant dependencies in the dataset


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Fitting the Multiple Linear Regression into the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Test the preformance of the model against the Test set
y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if(regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
        regressor_OLS.summary()
        return x
    
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
      
"""X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1) #adding the b0 constant (as a column of 1s) into the matrix of features
X_opt = X[:, [0, 1, 2, 3, 4, 5]] #ultimately will only contain the independent variables tha thave high impact on the profit
#Step2
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#Step3
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]] #ultimately will only contain the independent variables that thave high impact on the profit
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 4, 5]] #ultimately will only contain the independent variables that thave high impact on the profit
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]] #ultimately will only contain the independent variables that thave high impact on the profit
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3]] #contains the independent variables that have the highthigh impact on the profit
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()"""