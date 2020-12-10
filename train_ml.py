#!/usr/bin/env python
#-*- coding:UTF-8 -*-
#########################################################################
# File Name: train_dev.py
# Author: NLP_Team
# Mail: zhaozhenyu_tx@126.com
# Created Time: 10:46:34 2018-05-07
#########################################################################
import sys
import numpy as np
import sklearn

from sklearn import linear_model
from sklearn.svm import SVC 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


import pickle




def fit_save(x,y,para = {},model_type = 'LR',model_path = 'cur.model'):
    """
    训练与预测
    """
    if model_type=='LR':
        clf = linear_model.LogisticRegression(penalty=para['penalty'], C=para['C'], solver=para['solver'],class_weight='balanced')
    if model_type=='SVM':
        clf = SVC()
    if model_type=='GBDT':
        clf = GradientBoostingClassifier(learning_rate=para['learning_rate'],n_estimators=para['n_estimators'])#n_estimators=500,max_depth=50)
    if model_type == 'RF':
        clf = RandomForestClassifier(n_estimators = para['n_estimators'])#n_estimators = para['n_estimators'],n_jobs=para['n_jobs'])#),min_samples_leaf=3)
    if model_type == 'GNB':
        clf = GaussianNB()#priors=para['priors'])
    clf.fit(x,y)
    file=open(model_path,'wb')
    pickle.dump(clf,file,0)
    return clf

def getbestpara(x,y,para = {},model_type = 'LR',model_path = 'cur.best.model',cv = 5):
    if model_type=='LR':
        clf = linear_model.LogisticRegression()
    if model_type=='SVM':
        clf = SVC()
    if model_type=='GBDT':
        clf = GradientBoostingClassifier()
    param_test = para
    gsearch=GridSearchCV(estimator = clf,param_grid=param_test,cv = cv)
    gsearch.fit(x,y)
    
    clf = gsearch.best_estimator_
    return clf,gsearch
   
if __name__=='__main__':
    pass 

