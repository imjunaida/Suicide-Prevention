#DATA PREPROCESSING

#importing the lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
#reading the dataset in csv
acc1 = pd.read_csv('accidents.csv')
#describing the mean,count,std,min,max
acc1.describe()
#desplaying first 5 tuples
acc1.head()
#plotting graph b/w severity(D.V) Vs count
import seaborn as sns
sns.countplot(y = "severity" , data = acc1 )
plt.tight_layout()

#creating Dataframe with important features
pd.DataFrame( {"count": acc1["severity"].value_counts().values } , index = acc1["severity"].value_counts().index )
acc1= acc1.loc[acc1["severity"] >  1].loc[acc1["severity"] < 4]
acc1["month"] = acc1["time"].apply(lambda x:int(x[:2]))
acc1["day"] = acc1["time"].apply(lambda x:int(x[3:5]))
acc1["year"] = acc1["time"].apply(lambda x:int(x[6:8]))
acc1["hour"] =  acc1["time"].apply(lambda x: int(x[9:11]) if str(x)[15] == 'A' else 12 + int(x[9:11])  )
acc1["lon"] = acc1["lon"].apply(lambda x:abs(x))
#so that multinomialNB works (only with positive features)
#creating the date at the datetime format (easier to deal with)
acc1[ "date" ]= acc1[["month" , "day" ,"year"]].apply(lambda x:pd.datetime(month = x['month'] , day = x['day']  , year = 2000+x["year"]), axis = 1)
acc1["weekday"] =  acc1["date"].apply(lambda x:x.weekday())

#severity by hours
severity_by_hour = pd.crosstab(index = acc1["hour"] , columns = acc1["severity"] )
severity_by_hour = pd.DataFrame(severity_by_hour.values)
severity_by_hour["ratio"] = severity_by_hour.apply(lambda x:x[0]/float(x[1]) , axis = 1)
severity_by_hour.sort_values(by = "ratio")

# correlation heatmap
acc1_corr = acc1[["lat" , "lon" , "month" , "year" , "hour" , "weekday" , "severity"]]
correlation = acc1_corr.corr()
sns.heatmap(correlation)
plt.tight_layout()

# shifting to 0-1 values instead of 2-3 for DV & IN IV making the arrangement of coloumn
X = acc1[["month" , "hour" , "year", "weekday" ,"lon" , "lat"]]
y = acc1["severity"].apply(lambda x:x-2) 

#splitting dataset into training and test data set
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) 

#give the dimeensionality
X_train.shape
X_test.shape

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#KERNAL_SVM
# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making  Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#accuracy On the basis of test set
"""(5043+3765)correct pred/11960(X_test)
Out[7]: 0.7364548494983277
"""

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)
accuracies.mean()
accuracies.std()
"""accuracies.mean()
Out[4]: 0.742998352553542

accuracies.std()
Out[5]: 0.0004942339373970595
"""

# Applying Grid Search to find the best suited model and the best hyper-parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001, 0.0001]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 5,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


#RANDOM FOREST
#implimentation of Random Forest &ravel is used to place in 1-D
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100, criterion = 'entropy', random_state = 0)
random_forest.fit(X_train,y_train.values.ravel())

y_pred = random_forest.predict(X_test)
#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

"""accuracy acc to X_test
(6083+4562)/11960
Out[5]: 0.8900501672240803
"""

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = random_forest, X = X_train, y = y_train, cv = 5)
accuracies.mean()
accuracies.std()
accuracies.mean()
Out[7]: 0.8897034596375617

accuracies.std()
Out[8]: 0.0008890870323279448



#ANN
# Feature Scaling 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential#seq is req to initialize ann
from keras.layers import Dense#is used to build the layers

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu', input_dim = 6))

# Adding the second hidden layer#creating another hidden layer is optional since we are working on DL just create
classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 5, nb_epoch = 50)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#XG BOOST

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

#Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
