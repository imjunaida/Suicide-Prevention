# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:43:35 2019

@author: Junaid
"""


'''This is the regression part where we predict the suicide number given the list of features'''
import numpy as np
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('master2.csv')
dataset=dataset[dataset['suicides_no']>30]
X = dataset.iloc[:, [0,1,2,3,5,6,8]].values
y = dataset.iloc[:, 4].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:,0])
X[:, 2] = labelencoder.fit_transform(X[:,2])
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [3])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)

X = np.array(ct.fit_transform(X), dtype=np.float64)
X = X[:, 1:]
#X = np.append(arr = np.ones((X.shape[0],1)).astype(int), values = X, axis = 1)
#X=X[:, [1, 2, 3, 4,5,7,8,9,10]]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#### Feature Scaling
#X_train=X_train[0:5000,:]
#y_train=y_train[0:5000]
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#fitting linear regression in training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() 
regressor.fit(X_train, y_train)

#predicting test set results
y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

from sklearn.linear_model import Ridge
clf = Ridge(alpha=1.0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)


#Neuarl Network Implementation
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense( activation = 'relu',input_dim = 11, units = 18, kernel_initializer='uniform', ))

# Adding the second hidden layer
classifier.add(Dense(activation = 'relu',kernel_initializer='normal',units = 16))
classifier.add(Dense(activation = 'relu',kernel_initializer='normal',units = 12))
classifier.add(Dense(activation = 'relu',kernel_initializer='normal',units = 10))
classifier.add(Dense(activation = 'relu',kernel_initializer='normal',units = 8))
# Adding the output layer
classifier.add(Dense(kernel_initializer='normal',units = 1))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# Fitting the ANN to the Training set
y_test=np.reshape(y_test,(y_test.shape[0],1))
y_train=np.reshape(y_train,(y_train.shape[0],1))
classifier.fit(X_train, y_train, batch_size = 5, epochs = 50)
y_pred=classifier.predict(X_test)
classifier.save('Suicide_no.h5')
from keras.models import load_model
model = load_model('Suicide_no.h5')
y_pred = model.predict(X_test)
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
# evaluate model with standardized dataset

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(max_depth=8, random_state=0)
regr.fit(X_train, y_train)
y_pred=regr.predict(X_test)
# Building the optimal model using Backward Elimination
import statsmodels.api as sm
X = np.append(arr = np.ones((X.shape[0],1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 2, 3, 4, 5,7,8,9,10,11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 2, 3, 4, 5,6,8,9,10,11]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()


X_test=X_test[0:3270,:]
y_test=y_test[0:3270]
np.random.seed(1)
np.random.shuffle(X_test)
np.random.seed(1)
np.random.shuffle(y_test)
#splitting the data into k folds
Xfolds=np.split(X_test,10)
Yfolds=np.split(y_test,10) 
acc1=[]
for i in range(10):
    y_pred = model.predict(Xfolds[i])
    acc1.append(r2_score(Yfolds[i], y_pred))
acc2=[]
for i in range(10):
    y_pred = regr.predict(Xfolds[i])
    acc2.append(r2_score(Yfolds[i], y_pred))
from scipy import stats
stats.ttest_ind(acc1,acc2)