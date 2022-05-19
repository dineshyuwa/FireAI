#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
from keras.layers.advanced_activations import ELU
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[ ]:


num_classes=2
img_rows,img_cols=40,40
batch_size=60


# In[ ]:


train_data_dir='/Users/yuwarajalingamdinesh/Desktop/fire_dataset/train'
validation_data_dir='/Users/yuwarajalingamdinesh/Desktop/fire_dataset/val'

train_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows,img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator=validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_rows,img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)


# In[ ]:


model=Sequential()

#First CONV-ReLU Layer
model.add(Conv2D(512,(3,3),padding='same',input_shape=(img_rows,img_cols,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())

#Second CONV-ReLU Layer
model.add(Conv2D(512,(3,3),padding='same',input_shape=(img_rows,img_cols,3)))
model.add(Activation('relu'))
model.add(BatchNormalization())

#Max Pooling with Dropout
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

#3rd set of CONV-ReLU Layers
model.add(Conv2D(512,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

#4th Set of CONV-ReLU Layers
model.add(Conv2D(512,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

#Max Pooling with Dropout
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

#5th set of CONV-ReLU Layers
model.add(Conv2D(512,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

#6th Set of CONV-ReLU Layers
model.add(Conv2D(512,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

#Max Pooling with Dropout
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

#7th set of CONV-ReLU Layers
model.add(Conv2D(512,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

#8th Set of CONV-ReLU Layers
model.add(Conv2D(512,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())


#Max Pooling with Dropout
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

#9th set of CONV-ReLU Layers
model.add(Conv2D(512,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

#10th Set of CONV-ReLU Layers
model.add(Conv2D(512,(3,3),padding='same'))
model.add(Activation('relu'))
model.add(BatchNormalization())

#First set of FC or Dense Layers
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

# #Second set of FC or Dense Layers
model.add(Dense(512))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))


# #Third set of FC or Dense Layers
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

# #fourth set of FC or Dense Layers
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

# #fifth set of FC or Dense Layers
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

# #sixth set of FC or Dense Layers
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

# #seventh set of FC or Dense Layers
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

# #eighth set of FC or Dense Layers
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))


# #ninth set of FC or Dense Layers
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))


# #tenth set of FC or Dense Layers
model.add(Dense(256))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))

#Final Dense Layer
model.add(Dense(num_classes))
model.add(Activation("softmax"))

print(model.summary())


# In[ ]:


from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau

checkpoint=ModelCheckpoint("/Users/yuwarajalingamdinesh/Desktop/model.h5",
                           monitor="val_loss",
                           mode="min",
                           save_best_only=True,
                           verbose=1)

earlystop=EarlyStopping(monitor='val_loss',
                        min_delta=0.2,
                        patience=3,
                        verbose=1,
                        restore_best_weights=True)

reduce_lr=ReduceLROnPlateau(monitor='val_loss',
                            factor=0.2,
                            patience=3,
                            verbose=1,
                            min_delta=0.0001)

callbacks=[earlystop,checkpoint,reduce_lr]

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.0001),
              metrics=['accuracy'])


# In[ ]:


nb_train_samples=11677
nb_validation_samples=3476
epochs=20

history=model.fit_generator(
train_generator,
    steps_per_epoch=nb_train_samples//batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples//batch_size,
)


# In[ ]:


test_data_dir='/Users/yuwarajalingamdinesh/Desktop/fire_dataset/val'
test_datagen=ImageDataGenerator(rescale=1./255)

test_generator=test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_rows,img_cols),
    batch_size=batch_size,
    class_mode='categorical'
)


# In[ ]:


nb_test_samples=3476


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
Y_pred=model.predict_generator(test_generator,nb_test_samples//batch_size+1)
y_pred=np.argmax(Y_pred,axis=1)
print('Confusion Matrix')
print('classification Report')
classes=["fire","nofire"]
print(classification_report(test_generator.classes,y_pred,target_names=classes))


# In[ ]:


from matplotlib import pyplot as plt
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# 

# In[ ]:


from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

fpr, tpr, thresholds = metrics.roc_curve(test_generator.classes, y_pred, pos_label=0)

# Print ROC curve
plt.plot(fpr,tpr)
plt.show() 

# Print AUC
auc = np.trapz(tpr,fpr)
print('AUC:', auc)


# In[ ]:


import os
os.sys.path
from keras.preprocessing import image


# In[ ]:


import keras
import tensorflow as tf
from keras.models import load_model
from PIL import Image

model = load_model('/Users/yuwarajalingamdinesh/Desktop/models/best_models/model4.h5')


# In[ ]:


import cv2
import numpy as np
   
# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('/Users/yuwarajalingamdinesh/Desktop/Final_Year/archive-2/archive-2/fire13.mp4')
   
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video  file")
   
# Read until video is completed
while(cap.isOpened()):
      
  # Capture frame-by-frame
  ret,frame = cap.read()
  if ret == True:
   
    # Display the resulting frame
    cv2.imshow('Frame', frame)
        
    resized = cv2.resize(frame, (32,32), interpolation = cv2.INTER_AREA)
    
    frame=image.img_to_array(resized)
    
    frame = np.expand_dims(frame, axis=0)
    
    frame=frame*1./255
    
    predicted=model.predict(frame)
    print(predicted)
    
    if predicted[0][0]*100>50:
        print("fire detected")
    
    
   
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break
   
  # Break the loop
  else: 
    break
   
# When everything done, release 
# the video capture object
cap.release()
   
# Closes all the frames
cv2.destroyAllWindows()


# ### import cv2
# # Opens the Video file
# cap1= cv2.VideoCapture('/Users/yuwarajalingamdinesh/Desktop/Final_Year/archive-2/archive-2/nofire32.mp4')
# i=0
# while(cap1.isOpened()):
#     ret, frame = cap1.read()
#     if ret == False:
#         break
#     cv2.imwrite('/Users/yuwarajalingamdinesh/Desktop/non_fire_new/nonfire32'+str(i)+'.jpg',frame)
#     i+=1
#  
# cap1.release()
# cv2.destroyAllWindows()

# In[ ]:




