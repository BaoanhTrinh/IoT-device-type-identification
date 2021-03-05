#!/usr/bin/env python
# coding: utf-8

# In[108]:


import warnings
# warnings.filterwarnings('always')  # "error", "ignore", "always", "default", "module" or "once"
warnings.filterwarnings('ignore')


# In[109]:


import os
import pickle as pk
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.optimizers import RMSprop
from keras.utils import np_utils


# In[110]:


import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
import utils
from sklearn import preprocessing


# In[111]:


import keras
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Conv1D, GlobalMaxPooling1D,Reshape,GlobalAveragePooling1D
from keras import Model, layers
from keras import Input
from keras.layers import Convolution1D, ZeroPadding1D, MaxPooling1D, BatchNormalization, Activation, Dropout


# In[112]:


def is_eq(a):
    return lambda b: 1 if a == b else 0


# In[113]:


train = pd.read_csv(os.path.abspath('C:/Users/Admin/Desktop/work/IoT-device-type-identification-master/data/100_each_validation.csv'), low_memory=False)
validation = pd.read_csv(os.path.abspath('C:/Users/Admin/Desktop/work/IoT-device-type-identification-master/data/100_each_validation.csv'), low_memory=False)
test = pd.read_csv(os.path.abspath('C:/Users/Admin/Desktop/work/IoT-device-type-identification-master/data/100_each_test.csv'), low_memory=False)


# In[114]:


y_col = 'device_category'
# devs = train['device_category'].unique()
# devs[10]='unknown'
devs = ['security_camera', 'TV', 'smoke_detector', 'thermostat', 'water_sensor',
'watch' ,'baby_monitor' ,'motion_sensor', 'lights' ,'socket','unknown']


# In[115]:


x_train, y_train = utils.split_data(train,y_col)
x_test, y_test = utils.split_data(test,y_col)

y_train = np.asarray(y_train)
x_train = np.asarray(x_train)
y_test = np.asarray(y_test)
x_test = np.asarray(x_test)

lab_enc = preprocessing.LabelEncoder()
y_train = lab_enc.fit_transform(y_train)
y_test = lab_enc.fit_transform(y_test)
x_train = x_train.astype("float32")
y_train = y_train.astype("float32")
x_test = x_test.astype("float32")
y_test = y_test.astype("float32")


# In[116]:



x_train = utils.perform_feature_selection(x_train, y_train, 240)
x_test = utils.perform_feature_selection(x_test, y_test, 240)
print(x_train.shape)


# In[117]:


num_classes =10
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)
print('New y_train shape: ', y_train.shape)


# In[118]:


TIME_PERIODS=80 #80
num_sensors=3 # 3
input_shape=240 # 240

model_m = Sequential()
model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
model_m.add(Conv1D(100, 10, activation='relu', input_shape=(TIME_PERIODS, num_sensors)))
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(Conv1D(160, 10, activation='relu'))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(num_classes, activation='softmax'))
print(model_m.summary())


# In[119]:


callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]


# In[120]:


model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])


# In[121]:


BATCH_SIZE = 400
EPOCHS = 50


# In[122]:


history = model_m.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)


# In[123]:


score = model_m.evaluate(x_test, y_test, verbose=1)
print("\nAccuracy on test data: %0.2f" % score[1])
print("\nLoss on test data: %0.2f" % score[0])
#confusion_matrix
y_pred_test = model_m.predict(x_test)
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)
print(classification_report(max_y_test, max_y_pred_test))


# In[ ]:




