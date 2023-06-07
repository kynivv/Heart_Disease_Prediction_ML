import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import statsmodels.api as sm
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sn


# Data
data = pd.read_csv('framingham.csv')
data.drop(['education'], inplace= True, axis=1)
data.rename(columns= {'male':'Sex_male'}, inplace= True)


# Removing Null Values
data.dropna(axis=0, inplace=True)
#print(data.head(), data.shape)


# EDA
plt.figure(figsize=(7,5))
sn.countplot(x='TenYearCHD', data=data, palette='BuGn_r')

#plt.show()


# Splittong Data & Normalizing the Data
X = np.asarray(data[['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']])
y = np.asarray(data['TenYearCHD'])

X = preprocessing.StandardScaler().fit(X).transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)

#print ('Train set:', X_train.shape,  y_train.shape)
#print ('Test set:', X_test.shape,  y_test.shape)


# Model Training & Validation
from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
X_pred = logreg.predict(X_train)

print(f'Training Accuracy: {100*(1- (mae(X_pred, y_train)))}%')
print(f'Test Accuracy: {100*(1- (mae(y_pred, y_test)))}%')


