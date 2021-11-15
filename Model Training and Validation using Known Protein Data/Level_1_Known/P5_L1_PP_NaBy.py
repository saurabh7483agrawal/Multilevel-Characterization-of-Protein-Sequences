# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 17:04:45 2021

@author: lenovo
"""

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from collections import Counter
from imblearn.over_sampling import SMOTE 
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


df = pd.read_csv("Level_1_GP_NonGP_2684.csv")
X=df.iloc[:,1:311]
Y=df.iloc[:,311]

lda = LinearDiscriminantAnalysis(n_components=10)
Z = lda.fit(X, Y).transform(X)

sm = SMOTE(random_state=51)
Z_res, Y_res = sm.fit_resample(Z, Y)
print('Resampled dataset shape %s' % Counter(Y_res))

validation_size = 0.20
seed = 100
Z_res_train, Z_res_validation, Y_res_train, Y_res_validation = train_test_split(Z_res, Y_res, test_size=validation_size, random_state=seed)                    ## shuffle and split training and test sets
scoring = 'accuracy'

gnb = GaussianNB()
gnb1 = gnb.fit(Z_res_train, Y_res_train)
gnb_prediction = gnb1.predict(Z_res_validation) 
print("Adaboost ::",confusion_matrix(Y_res_validation,gnb_prediction))
print("PRECISION SCORE::",precision_score(Y_res_validation,gnb_prediction, average="macro"))
print("RECALL SCORE::",recall_score(Y_res_validation,gnb_prediction, average="micro"))
print("F1_SCORE::",f1_score(Y_res_validation,gnb_prediction, average="weighted"))
print("Accuracy::",accuracy_score(Y_res_validation,gnb_prediction))
print('Classification Report::', classification_report(Y_res_validation, gnb_prediction))














