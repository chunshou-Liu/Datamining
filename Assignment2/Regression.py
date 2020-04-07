# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 00:00:12 2018

@author: Susan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error,r2_score

# 1-a Read file and select data from 2018/10/01-12/31
#data = pd.read_excel('E:/DB/HW2/106年新竹站_20180309.xls')
#data = data[data["日期"].between("2017/10/01","2017/12/31")].drop("測站",axis=1)
#data = data.reset_index().drop('index',axis=1)
#data.to_excel("E:/DB/HW2/preprocess_data.xls")

# open the preprocess_data 
data = pd.read_excel('E:/DB/HW2/preprocess_data.xls')

# 1-b,1-c 
# 1-c replace NR with 0
data = data.replace('NR',0)
data = data.fillna('#')
data = data.groupby(['測項'])
data = [data.get_group(x) for x in data.groups]
data2 = data.copy()

# 1-b complete the loss data
for i in range(18):
    flat,df = [],[]
    flat = data[i].drop(columns = ['日期','測項'])
    df = data[i].loc[:, data[i].columns.intersection(['日期','測項'])].reset_index().drop('index',axis=1)
    
    flat = flat.values.flatten()
    for j in range(2208):
        if (isinstance(flat[j], str)):
            start = flat[j-1]
            startidx = j
            j = j +1
            while (isinstance(flat[j], str)):
                j = j+1
            end = flat[j]
            endidx = j-1
            for k in (startidx,endidx):
                flat[k] = (start+end)/2
    flat = pd.DataFrame(np.asarray([flat]).reshape(92,24))
    data2[i] = pd.concat([df, flat],axis=1)

# 1-d split data into training and testing set
data = data2.copy()
data2 = pd.concat([data2[i] for i in range(0,18)],axis=0)

data2['日期'] = pd.to_datetime(data2['日期'])

train = data2[(data2['日期'] >= '2017/10/01 00:00:00') & (data2['日期'] <= '2017/11/30 00:00:00')]
test = data2[(data2['日期'] >= '2017/12/01 00:00:00') & (data2['日期'] <= '2017/12/31 00:00:00')]

# 1-e make data into time series
A = list(data2.測項.unique())
train = train.groupby(['測項'])
train = [train.get_group(x) for x in train.groups]
test = test.groupby(['測項'])
test = [test.get_group(x) for x in test.groups]

for i in range(18):
    flat_train,flat_test = [],[]    
    train[i] = list(pd.DataFrame(train[i]).drop(columns = ['日期','測項']).values.flatten())
    test[i] = list(pd.DataFrame(test[i]).drop(columns = ['日期','測項']).values.flatten())

train = pd.concat([pd.DataFrame(train[i] for i in range(18))],axis=0)
test = pd.concat([pd.DataFrame(test[i] for i in range(18))],axis=0)
test.index = A
train.index = A

# 2-a setting train_y & test_y
test_y = test.iloc[9][6:1464]
train_y =  train.iloc[9][6:1464]

window = 6
# 2-b.1 train_x_pm & test_x_pm
train_x_pm = np.array([train.iloc[9][j:j+window].tolist() for j in range(len(train.columns) - window)]).reshape(-1,6)
test_x_pm = np.array([test.iloc[9][j:j+window].tolist() for j in range(len(test.columns) - window)]).reshape(-1,6)


# 2-b.2 train_x & test_x
train_x,test_x = [],[]
for i in range(len(train.columns) - window):
    train_x.append(train.values[:,i:i+window])
train_x = np.asarray(train_x)

for i in range(len(test.columns) - window):
    test_x.append(test.values[:,i:i+window])
test_x = np.asarray(test_x)


# 2-c.1 Linear regression
# Create the regression object
regr = linear_model.LinearRegression()
regr_pm = linear_model.LinearRegression()
# Train the model using train_x 
regr.fit(train_x.reshape((-1,6*18)), train_y)
train_x_pm = list(train_x_pm)

regr_pm.fit(train_x_pm, train_y)

# Make predictions using the testing set
pred_y = regr.predict(test_x.reshape((-1,6*18)))
pred_y_pm = regr_pm.predict(test_x_pm)


# 2-d.1 Linear regression :The mean absolute error
print("Linear - Mean absolute error: %.2f"
      % mean_absolute_error(test_y, pred_y))
print("Linear_pm - Mean absolute error: %.2f"
      % mean_absolute_error(test_y, pred_y_pm))

# 2-c.2 Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(max_depth=15, random_state=0,n_estimators=100)
regr.fit(train_x.reshape(-1,6*18), train_y)

regr_pm = RandomForestRegressor(max_depth=15, random_state=0,n_estimators=100)
regr_pm.fit(train_x_pm, train_y)

pred_y = regr.predict(test_x.reshape(-1,6*18))
pred_y_pm = regr_pm.predict(test_x_pm)
# 2-d.2 Random Forest Regression :The mean absolute error
print("Random Forest - Mean absolute error: %.2f"
      % mean_absolute_error(test_y, pred_y))
print("Random Forest_pm - Mean absolute error: %.2f"
      % mean_absolute_error(test_y, pred_y_pm))
