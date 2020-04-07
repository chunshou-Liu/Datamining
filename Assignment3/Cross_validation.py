# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 18:41:58 2018

@author: Susan
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# 1 Read file 
data = pd.read_csv('E:/DM/HW3/data.csv')

# 2-1 replace all tha space in head and tail
data = data.replace('^\s+','',regex=True).replace('\s+$','',regex=True)

# 2-2 drop data with ?
data = data.replace('?',np.nan).dropna()

# 2-3 set label:income>50 as 1
data = pd.get_dummies(data).drop(['income_>50K'],axis=1)

# 2-4 set trainning set and testing set
x = data.values[:,:-1]
y = data.values[:,-1].reshape(-1,1)         

# 3 Define function
def Cross_validation(k, data):
     #設定subset size 即data長度/k
     group = np.mod(np.arange(len(data)), k).reshape(-1,1)
     #設定Accuracy初始值
     Accuracy = 0
     for i in range(k):
        #設定testing set與training set的資料起始點與結束點
        trainidx = np.where(group!=i)[0] 
        testidx = np.where(group==i)[0]
        #例如資料有100筆，testing set在本次iteration取第1到10筆，則training set為第11到100筆；下次testing set為11~20，training set為21~100 & 1~10
        train_x, train_y = x[trainidx], y[trainidx]
        test_x, test_y =  x[testidx], y[testidx]
        # GradientBoostingClassifier
        clf = GradientBoostingClassifier().fit(train_x, train_y)
        print(clf.score(test_x, test_y))           
        #利用training set建立模型，testing set計算出Accuracy累加
        Accuracy = Accuracy + clf.score(test_x, test_y)    
     return Accuracy/k
 
Ans = Cross_validation(10,data)