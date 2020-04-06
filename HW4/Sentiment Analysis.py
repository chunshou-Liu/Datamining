# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 21:27:51 2018

@author: Susan
"""
import pandas as pd
from keras.preprocessing import sequence

# 1 read files split by \n (data has no header)
training = pd.read_csv('training_label.txt', sep = '\n', header = None)
testing = pd.read_csv('testing_label.txt', sep = '\n', header = None)

# 1-a split data into train and test by +++$+++ & #####
train = pd.DataFrame(training[0].str.split(' \+\+\+\$\+\+\+ ').tolist(), columns=['y', 'x'])
test =  pd.DataFrame(testing[0].str.split('#####').tolist(), columns=['y', 'x'])

# 1-b set test and train 
train_x = train['x'] 
train_y = train['y']
test_x = test['x']
test_y = test['y']

# 2-1 Tokenize the data
from keras.preprocessing.text import Tokenizer
token = Tokenizer(7000)
token.fit_on_texts(train_x)

x_train_seq = token.texts_to_sequences(train_x)
x_test_seq = token.texts_to_sequences(test_x)

# 2-2 set max length of data
x_train = sequence.pad_sequences(x_train_seq, maxlen=100)
x_test = sequence.pad_sequences(x_test_seq, maxlen=100)
#%%
from s
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, 
import matplotlib.pyplot as plt

modelRNN = Sequential()  #建立模型
#Embedding層將「數字list」轉換成「向量list」
modelRNN.add(Embedding(output_dim=4,   #輸出的維度是32，希望將數字list轉換為32維度的向量
     input_dim=7000,  #輸入的維度是3800，也就是我們之前建立的字典是3800字
     input_length=100)) #數字list截長補短後都是380個數字

# 3-1 set RNN model 
#modelRNN.add(Dropout(0.7))      	             #隨機在神經網路中放棄70%的神經元，避免overfitting
modelRNN.add(SimpleRNN(units=16))                #建立16個神經元的RNN層
modelRNN.add(Dense(units=256,activation='relu')) #建立256個神經元的隱藏層，ReLU激活函數
#modelRNN.add(Dropout(0.7))
modelRNN.add(Dense(units=1,activation='sigmoid'))#建立一個神經元的輸出層，Sigmoid激活函數
modelRNN.summary()

# 3-2 Complie the model and fit data 
modelRNN.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) 
#train_history = modelRNN.fit(x_train,train_y, epochs=20, batch_size=1024,verbose=1,validation_data=[x_test, test_y])
train_history = modelRNN.fit(x_train,train_y, epochs=20, batch_size=1024,verbose=1,validation_split=0.2)
scores = modelRNN.evaluate(x_test, test_y,verbose=1)

# 3-3 Plot Accuracy and Loss
plt.plot(train_history.history['acc'])
plt.plot(train_history.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.title('RNN model accuracy')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('RNN model loss')
plt.legend(['train', 'validation'], loc='upper left')
#%%
#LSTM
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers import CuDNNLSTM
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt
#from keras.layers.recurrent import CuDNNLSTM

modelLSTM = Sequential() #建立模型
modelLSTM.add(Embedding(output_dim=4, input_dim = 7000 , input_length = 100)) 
#輸出的維度是32，希望將數字list轉換為32維度的向量
#輸入的維度是3800，也就是我們之前建立的字典是3800字
#數字list截長補短後都是380個數字

modelLSTM.add(Dropout(0.7)) #隨機在神經網路中放棄20%的神經元，避免overfitting
modelLSTM.add(CuDNNLSTM(32))                        #建立32個神經元的LSTM層
modelLSTM.add(Dense(units=256,activation='relu'))   #建立256個神經元的隱藏層
modelLSTM.add(Dropout(0.7))
modelLSTM.add(Dense(units=1,activation='sigmoid'))  #建立一個神經元的輸出層
 
modelLSTM.summary()
modelLSTM.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) 
train_lstm_history = modelLSTM.fit(x_train,train_y, epochs=20, batch_size=1024,verbose=1,validation_split=0.2)
#train_lstm_history = modelLSTM.fit(x_train,train_y, epochs=20, batch_size=1024,verbose=1,validation_data=[x_test, test_y])

scores = modelLSTM.evaluate(x_test, test_y,verbose=1)
#%%
plt.plot(train_lstm_history.history['acc'])
plt.plot(train_lstm_history.history['val_acc'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.title('LSTM model accuracy')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
#%%
plt.plot(train_lstm_history.history['loss'])
plt.plot(train_lstm_history.history['val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('LSTM model loss')
plt.legend(['train', 'validation'], loc='upper left')