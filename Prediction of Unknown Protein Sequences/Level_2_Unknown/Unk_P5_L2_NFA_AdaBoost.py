# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 13:07:08 2021

@author: Indian
"""
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from collections import Counter
from imblearn.over_sampling import SMOTE 



df = pd.read_csv("Train_Level_2_Path_NonPath_GP.csv")
X=df.iloc[:,1:311]
Y=df.iloc[:,311]

lda = LinearDiscriminantAnalysis(n_components=1)
Z = lda.fit(X, Y).transform(X)

sm = SMOTE(random_state=51)
Z_res, Y_res = sm.fit_resample(Z, Y)
print('Resampled dataset shape %s' % Counter(Y_res))

validation_size = 0.10
seed = 10
Z_res_train, Z_res_validation, Y_res_train, Y_res_validation = train_test_split(Z_res, Y_res, test_size=validation_size, random_state=seed)                    ## shuffle and split training and test sets
scoring = 'accuracy'

ada = AdaBoostClassifier(n_estimators=100,learning_rate=1,random_state = 0)
ada1 = ada.fit(Z_res_train, Y_res_train)
ada_prediction = ada1.predict(Z_res_validation) 
print("Adaboost ::",confusion_matrix(Y_res_validation,ada_prediction))
print("PRECISION SCORE::",precision_score(Y_res_validation,ada_prediction, average="macro"))
print("RECALL SCORE::",recall_score(Y_res_validation,ada_prediction, average="micro"))
print("F1_SCORE::",f1_score(Y_res_validation,ada_prediction, average="weighted"))
print("Accuracy::",accuracy_score(Y_res_validation,ada_prediction))

test = pd.read_csv("Unk_NFA_Test_Level_2_Path_NonPath_GP.csv")

ZN = lda.transform(test)

pred = ada1.predict(ZN)

pred1 = ada1.predict_proba(ZN)
