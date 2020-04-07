# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 11:17:01 2018

@author: Susan
"""
import os
os.environ["PATH"] += os.pathsep + 'D:/Anaconda3/Library/bin/graphviz'

import pandas as pd 
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,precision_score
import graphviz
from IPython.display import Image 
import pydotplus

# 1 Read file 
data = pd.read_csv('E:/DB/character-deaths.csv')

# 2-1 data complition：set 0
data['Book of Death'] = data['Book of Death'].fillna(0) 
data['Book Intro Chapter'] = data['Book Intro Chapter'].fillna(0) 
data = data.drop(columns = ['Death Year','Death Chapter'])
# check nan number of data
#data['Death Year'].isna().sum()
#data['Book of Death'].isna().sum()
#data['Death Chapter'].isna().sum()

# check out 'Death Year' == na but Book of 'Death' != nan
#data[np.logical_and(pd.isna(data['Death Year']) , pd.notna(data['Book of Death'])) == True]

# 2-2 data complition：set 1
data['Book of Death'][data['Book of Death'] > 0] = 1

# 2-3 adding columns by 'Allegiances'
data = pd.concat([data,pd.get_dummies(data['Allegiances'])],axis =1 ,join='outer')
data = data.drop(columns = ['Allegiances'])

# 2-4 divid into training set  and testing set 
train, test = train_test_split(data, test_size=0.25,random_state = 0)
x_train = train.drop( ['Book of Death' ,'Name'],axis=1 )
y_train = train['Book of Death']
x_test = test.drop( ['Book of Death' ,'Name'],axis=1 )
y_test = test['Book of Death']

# 3 DecisionTreeClassifier of scikit-learn 
clf = tree.DecisionTreeClassifier(max_depth = 10 , random_state = 0)
clf = clf.fit(x_train, y_train)


# 4 Make the confusion matrix and calculate Precision, Recall, Accuracy
y_pred = clf.predict(x_test)
y_true = y_test
print('Confusion matrix：' + '\n' + str(confusion_matrix(y_true,y_pred))) 
print('Precision：' + str(precision_score(y_true,y_pred)))
print('Recall：' + str(recall_score(y_true, y_pred)))
print('Accuracy：'+ str(accuracy_score(y_true, y_pred)))

# 5 Generate the grap of the decisiontree
dot_data = tree.export_graphviz(clf, out_file=None, 
                                feature_names=x_train.columns,
                                class_names=['0','1'],
                                filled=True)

graph = graphviz.Source(dot_data)
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())
graph.write_pdf("DMHW1_0753407_DecisionTree.pdf")