# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 16:14:01 2021

@author: Indian
"""
import pandas as pd 
#import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
#from sklearn.metrics import classification_report
from collections import Counter
from imblearn.over_sampling import SMOTE 

df = pd.read_csv("Train_Level_3_StaAuC1_ClosC2_StrPnC3.csv")
X=df.iloc[:,1:311]
Y=df.iloc[:,311]

lda = LinearDiscriminantAnalysis(n_components=1)
Z = lda.fit(X, Y).transform(X)

sm = SMOTE(random_state=51)
Z_res, Y_res = sm.fit_resample(Z, Y)
print('Resampled dataset shape %s' % Counter(Y_res))

validation_size = 0.10
seed = 100
Z_res_train, Z_res_validation, Y_res_train, Y_res_validation = train_test_split(Z_res, Y_res, test_size=validation_size, random_state=seed)                    ## shuffle and split training and test sets
scoring = 'accuracy'


dtc = tree.DecisionTreeClassifier(random_state = 1)

dtc1 = dtc.fit(Z_res_train, Y_res_train)
ada_prediction = dtc1.predict(Z_res_validation) 
print("Adaboost ::",confusion_matrix(Y_res_validation,ada_prediction))
print("PRECISION SCORE::",precision_score(Y_res_validation,ada_prediction, average="macro"))
print("RECALL SCORE::",recall_score(Y_res_validation,ada_prediction, average="micro"))
print("F1_SCORE::",f1_score(Y_res_validation,ada_prediction, average="weighted"))
print("Accuracy::",accuracy_score(Y_res_validation,ada_prediction))


test = pd.read_csv("Unk_Test_Level_3_StaAuC1_ClosC2_StrPnC3.csv")

ZN = lda.transform(test)

pred = dtc.predict(ZN)

pred1 = dtc.predict_proba(ZN)