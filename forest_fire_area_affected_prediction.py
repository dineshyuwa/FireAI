#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


#import dataset and split to target and inputs
df=pd.read_csv("/Users/yuwarajalingamdinesh/Desktop/Final_Year/forestfires.csv")

df = df.drop('month',axis = 1)
df=df.drop('day',axis=1)
df=df.drop('X',axis=1)
df=df.drop('Y',axis=1)
df=df.drop('rain',axis=1)
df=df.drop('FFMC',axis=1)
df=df.drop('DMC',axis=1)
df=df.drop('DC',axis=1)
df=df.drop('ISI',axis=1)
print(df)


# In[ ]:


X=df.iloc[:,:3].values
Y=df.iloc[:,3].values

from sklearn.preprocessing import MinMaxScaler

#normailizing the train and test datasets
scaler=MinMaxScaler()
scaler.fit(Y.reshape(-1,1))
Y_train=scaler.transform(Y.reshape(-1,1))
X_train=np.array(X)
Y_train=np.array(Y_train)
print(X_train)


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout

#creatin model with one input layer one hidden layer and one output layer
model=Sequential()
model.add(Dense(units=50,activation='relu',input_dim=3))
model.add(Dense(units=150,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=100,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=1))
#
#

#compiling with 100 epochs with batchsize 100
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
model.fit(X_train,Y_train,batch_size=90,epochs=120)


# In[ ]:


y_pred=model.predict(X_train)
true_predictions=scaler.inverse_transform(y_pred)


# In[ ]:


print(y_pred)


# In[ ]:


print(Y)


# In[ ]:


print(Y_train-y_pred)


# In[ ]:




prediction=model.predict(np.array([[21.2,70,6.7]]))
true_prediction=scaler.inverse_transform(prediction)


# In[ ]:


true_prediction[0][0]


# In[ ]:


model.save("/Users/yuwarajalingamdinesh/Desktop/models/area_burnt/area_burnt_model6.h5")


# In[ ]:




