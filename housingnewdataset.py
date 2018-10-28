# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:59:48 2017

@author: hp
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm

from sklearn.model_selection import cross_val_predict
from sklearn.cross_validation import train_test_split

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")

# special matplotlib argument for improved plots
from matplotlib import rcParams

#from sklearn.datasets import load_boston
boston = pd.read_csv('C:\\Users\\hp\\Downloads\\ex3x.csv')

print(boston.keys())
print(boston.shape)
print(boston.x1)
print(boston.y)

bos = pd.DataFrame(boston.x1)
print(bos.head())

print(boston.y.shape)
bos['PRICE'] = boston.y
print(bos.head())
print(bos.describe())
X = bos.drop('PRICE', axis = 1)
Y = bos['PRICE']

X_train, X_test, Y_train, Y_test = sklearn.cross_validation.train_test_split(X, Y, test_size = 0.33, random_state = 5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train, Y_train)

Y_pred = lm.predict(X_test)

plt.scatter(Y_test, Y_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")