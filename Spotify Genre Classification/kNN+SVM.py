# Josh Morgan
# kNN and SVM classifiers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

dataset = pd.read_csv("train.csv")

#Use this to filter features that you might not want
feature_columns = ['Popularity', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                    'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_in min/ms', 'time_signature']

#Feel free to adjust the test_size, random_state, and columns that should be used.
X_train, X_test, y_train, y_test = train_test_split(dataset[feature_columns], dataset['Class'], 
                                                    test_size=0.2, random_state=42)

#If the model that you are using doesn't split the training set into a validation set, here is a basic holdout set
#for validation that is commented out by default
#X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Combining Alt Music, Indie Alt, and Rock
y_train = y_train.replace(to_replace=[1, 6], value=10)
y_test = y_test.replace(to_replace=[1, 6], value=10)

# Imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp_mean = IterativeImputer(random_state=42)
imp_mean.fit(X_train)

X_train = imp_mean.transform(X_train)
X_test = imp_mean.transform(X_test)

# Scaling Data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier


kNN = KNeighborsClassifier()

#Searching for best parameter
params = {'n_neighbors':[1,5,10,20,30,40,50,60,70,80,90,100]}
kNN_GS = GridSearchCV(estimator=kNN, param_grid=params, scoring='accuracy', cv=5)

kNN_GS.fit(X_train, y_train)

score = kNN_GS.best_score_
params = kNN_GS.best_params_
print('kNN Accuracy:',score)
print('Best Parameters:',params)

#Narrowing the scope of the search
params = {'n_neighbors':[28,29,30,31,32,33,34]}
kNN_GS = GridSearchCV(estimator=kNN, param_grid=params, scoring='accuracy', cv=5)

kNN_GS.fit(X_train, y_train)

score = kNN_GS.best_score_
params = kNN_GS.best_params_
print('kNN Accuracy:',score)
print('Best Parameters:',params)

#Running the most accurate parameter for model
kNN = KNeighborsClassifier(31)
kNN.fit(X_train,y_train)
y_preds = kNN.predict(X_test)
print('Highest Accuracy for kNN:', metrics.accuracy_score(y_test,y_preds))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Create confusion matrix 
predictions = kNN.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=kNN.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=kNN.classes_)
disp.plot()
plt.show()


#SVM
from sklearn import svm

svc = svm.SVC()
svc.fit(X_train,y_train)

y_preds = svc.predict(X_test)
print('SVM Accuracy:',metrics.accuracy_score(y_test,y_preds))

#params = {'C': np.logspace(-5, 5, 13),
#        'gamma': np.logspace(-5, 5, 13),
#         'kernel' : ['rbf']}

#svm_RGS = RandomizedSearchCV(svc, params, 
#                             n_iter =30, cv=3)

#svm_RGS.fit(X_train,y_train)
#svm_RGS.best_params_
#svm_RGS.best_score_


#score = svm_RGS.best_score_
#params = svm_RGS.best_params_
#print('SVM Accuracy:',score)
#print('Best Parameters:',params)

# Best Parameters: {'kernel': 'rbf', 'gamma': 0.0031622776601683794, 'C': 2154.4346900318865}

svc = svm.SVC(C=2154.4346900318865, kernel='rbf',gamma= 0.0031622776601683794)
svc.fit(X_train,y_train)

y_preds = svc.predict(X_test)
print('SVM Accuracy:',metrics.accuracy_score(y_test,y_preds))

predictions = svc.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=svc.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=svc.classes_)
disp.plot()
plt.show()
