# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 17:16:04 2021

@author: lenovo
"""

import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from collections import Counter
from imblearn.over_sampling import SMOTE 


#df = pd.read_csv("Gram_Negative.csv")
df = pd.read_csv("Level_4_a_Path_StaAu_PSL_1_2_3.csv")
X=df.iloc[:,1:311]
Y=df.iloc[:,311]

validation_size = 0.20
seed = 50

lda = LinearDiscriminantAnalysis(n_components=10)
Z = lda.fit(X, Y).transform(X)

sm = SMOTE(random_state=50)
Z_res, Y_res = sm.fit_resample(Z, Y)
print('Resampled dataset shape %s' % Counter(Y_res))


Z_res_train, Z_res_validation, Y_res_train, Y_res_validation = train_test_split(Z_res, Y_res, test_size=validation_size, random_state=seed)                    ## shuffle and split training and test sets
scoring = 'accuracy'


grBoosting = GradientBoostingClassifier(learning_rate=0.01, min_samples_split=100, 
                                        min_samples_leaf=50,max_depth=8,max_features='sqrt',
                                        subsample=1,random_state=0)
grBoosting1 = grBoosting.fit(Z_res_train,Y_res_train)
grBoosting_prediction=grBoosting1.predict(Z_res_validation)
print("GradientBoosting ::\n",confusion_matrix(Y_res_validation,grBoosting_prediction))
print("PRECISION SCORE::",precision_score(Y_res_validation,grBoosting_prediction, average="macro"))
print("RECALL SCORE::",recall_score(Y_res_validation,grBoosting_prediction, average="macro")) 
print("F1_SCORE::",f1_score(Y_res_validation,grBoosting_prediction, average="weighted"))
print("Accuracy::",accuracy_score(Y_res_validation,grBoosting_prediction))

#test = pd.read_csv("GN_Test.csv")

#ZN = lda.transform(test)

#pred = grBoosting1.predict(ZN)