#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# first of all import the socket library
import socket

# next create a socket object
s = socket.socket()
print ("Socket successfully created")

 
# reserve a port on your computer in our
# case it is 12345 but it can be anything
port = 1234            
 
# Next bind to the port
# we have not typed any ip in the ip field
# instead we have inputted an empty string
# this makes the server listen to requests
# coming from other computers on the network
s.bind(('169.254.154.230', port))        
print ("socket binded to %s" %(port))
 
# put the socket into listening mode
s.listen(5)    
print ("socket is listening")

# Establish connection with client.
c, addr = s.accept()    
print ('Got connection from', addr )
 
# a forever loop until we interrupt it or
# an error occurs
while True:
    
  data=c.recv(500)
  print(data)


# In[ ]:


from flask import Flask
from flask_socketio import SocketIO, send

app=Flask(__name__)
app.config["SECRET_KEY"]='mysecret'

socketIo=SocketIO(app,cors_allowed_origins="*")
app.debug=True
app.host="localhost"

@socketIo.on("message")
def handleMessage(msg):
    print(msg)
    send("hello",broadcast=True)
    return None

if __name__=='__main__':
    socketIo.run(app)


# In[ ]:


pip install nest-asyncio


# In[ ]:


import asyncio
import random
import datetime
import websockets
import json
import nest_asyncio
import socket
import pandas as pd
import keras
import tensorflow as tf
import numpy as np
from keras.models import load_model
nest_asyncio.apply()

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

X=df.iloc[:,:3].values
Y=df.iloc[:,3].values

from sklearn.preprocessing import MinMaxScaler

#normailizing the train and test datasets
scaler=MinMaxScaler()
scaler.fit(Y.reshape(-1,1))
Y_train=scaler.transform(Y.reshape(-1,1))

model = load_model('/Users/yuwarajalingamdinesh/Desktop/models/area_burnt/area_burnt_model6.h5')

# next create a socket object
s = socket.socket()
print ("Socket successfully created")

 
# reserve a port on your computer in our
# case it is 12345 but it can be anything
port = 1234            
 
# Next bind to the port
# we have not typed any ip in the ip field
# instead we have inputted an empty string
# this makes the server listen to requests
# coming from other computers on the network
s.bind(('192.168.8.172', port))        
print ("socket binded to %s" %(port))
 
# put the socket into listening mode
s.listen(5)    
print ("socket is listening")

# Establish connection with client.
c, addr = s.accept()    
print ('Got connection from', addr )
 
# a forever loop until we interrupt it or
# an error occurs

async def handler(websocket, path):
    while True:
        data=c.recv(500)
        data=data.decode("utf-8")
        data=json.loads(data)[0]
        print(data)
        temperature=data["temperature"]
        relativeHumidity=data["humidity"]
        wind_speed=data["wind_speed"]
        prediction=model.predict(np.array([[temperature,relativeHumidity,wind_speed]]))
        true_prediction=scaler.inverse_transform(prediction)
        if true_prediction[0][0]<0:
            true_prediction=0
        data["Damaged_Area"]=true_prediction[0][0]
        print(str(data)[1:-1])
        await websocket.send(str(data)[1:-1])
        await asyncio.sleep(0.5)

start_server = websockets.serve(handler, "127.0.0.1", 5000)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




