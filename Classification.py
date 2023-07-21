# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:20:54 2019

@author: Junaid
"""

import numpy as np
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('master2.csv')
dataset=dataset[dataset['suicides_no']>30]
X = dataset.iloc[:, [0,1,2,4,5,6,8]].values
y = dataset.iloc[:, 3].values
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:,0])
X[:, 2] = labelencoder.fit_transform(X[:,2])
X[:, 1] = labelencoder.fit_transform(X[:,1])
# encode class values as integers
from keras.utils import np_utils
encoder = LabelEncoder()
encoder.fit(y)
y= encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
y = np_utils.to_categorical(y)
#y=y[:,1:]
#X = X[:, 1:]
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
# Fitting Neural Network to the Training set
##Neuarl Network Implementation
#from keras.models import Sequential
#from keras.layers import Dense
## Initialising the ANN
#classifier = Sequential()
## Adding the input layer and the first hidden layer
#classifier.add(Dense( activation = 'relu',input_dim = 7, units = 18, kernel_initializer='uniform', ))
#
## Adding the second hidden layer
#classifier.add(Dense(activation = 'relu',kernel_initializer='normal',units = 12))
#classifier.add(Dense(activation = 'relu',kernel_initializer='normal',units = 10))
#classifier.add(Dense(activation = 'relu',kernel_initializer='normal',units = 8))
#classifier.add(Dense(activation = 'relu',kernel_initializer='normal',units = 8))
#classifier.add(Dense(activation = 'relu',kernel_initializer='normal',units = 8))
## Adding the output layer
#classifier.add(Dense(activation = 'softmax',kernel_initializer='normal',units = 6))
#
## Compiling the ANN
#classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#
## Fitting the ANN to the Training set
#classifier.fit(X_train, y_train, batch_size = 8, epochs = 300)
# training a linear SVM classifier 
from sklearn.neighbors import KNeighborsClassifier 
classifier = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train)
# Fitting Decision Tree to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0,min_samples_leaf=1)
classifier.fit(X_train, y_train)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 300, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_score = classifier.predict_proba(X_test)
npa = np.asarray(y_score, dtype=np.float32)
y_score=npa[:,:,1]
from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
print(classifier.score(X_test,y_test))
from sklearn.metrics import roc_curve, auc
from scipy import interp
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(0,6):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i],y_score[i,:] )
    roc_auc[i] = auc(fpr[i], tpr[i])
import matplotlib.pyplot as plt
plt.figure()
lw = 2
plt.plot(fpr[0], tpr[0], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
plt.plot(fpr[1], tpr[1], color='pink',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot(fpr[2], tpr[2], color='black',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot(fpr[3], tpr[3], color='magenta',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[3])
plt.plot(fpr[4], tpr[4], color='brown',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[4])
plt.plot(fpr[5], tpr[5], color='green',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[5])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic ')
plt.legend(loc="lower right")
plt.show()

#Statistical Significance test
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
    acc1.append(classifier.score(Xfolds[i],Yfolds[i]))
acc2=[]
for i in range(10):
    acc2.append(classifier.score(Xfolds[i],Yfolds[i]))
from scipy import stats









