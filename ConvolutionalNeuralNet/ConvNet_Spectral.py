# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:06:36 2019

@author: Avishek Paul
"""

import os
from os import walk
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.models import load_model
import pandas as pd

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam,SGD
from keras.layers.normalization import BatchNormalization
from layer_utils import AttentionLSTM  # Location in C:\Users\SakataWoolley\Desktop\LSTM-FCN-master\utils

import matplotlib.pyplot as plt

img_width = 128;
img_height = 128;
batch_size_num = 64;

#path_train = "F:\data_for_avishek\LSTMGeneric2\Spectral_Images\Train\\"
#path_test = "F:\data_for_avishek\LSTMGeneric2\Spectral_Images\Test\\"
path_train = "F:\data_for_avishek\LSTMGeneric3\Spectral_Images\Train\\"
path_test = "F:\data_for_avishek\LSTMGeneric3\Spectral_Images\Test\\"

def create_model_matlab(p,input_shape,decay_rate):
     numF = 12;
     model = Sequential()
     
     model.add(Conv2D(numF, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
     model.add(BatchNormalization())
     model.add(MaxPooling2D(pool_size=(3, 3),strides=2,padding='same'))
     
     model.add(Conv2D(2*numF, (3, 3), padding='same', activation='relu'))
     model.add(BatchNormalization())
     model.add(MaxPooling2D(pool_size=(3, 3),strides=2,padding='same'))
        
     model.add(Conv2D(4*numF, (3, 3), padding='same', activation='relu'))
     model.add(BatchNormalization())
     model.add(MaxPooling2D(pool_size=(3, 3),strides=2,padding='same'))
     
#     model.add(Conv2D(4*numF, (3, 3), padding='same', activation='relu'))
#     model.add(BatchNormalization())
     model.add(Conv2D(4*numF, (3, 3), padding='same', activation='relu'))
     model.add(BatchNormalization())
     
     model.add(MaxPooling2D(pool_size=(1, 13)))
     
     model.add(Flatten())
     model.add(Dropout(p))
     # Fully connection
     model.add(Dense(4*numF, activation='softmax'))
     
     model.add(Dense(1, kernel_initializer='normal',activation='sigmoid'))
     
     sgd = SGD(lr=1e-3,momentum=0.95,decay = decay_rate,nesterov=True)
     metrics=['accuracy']
     model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=metrics)
          
     return model
          
#def create_model(p, input_shape):
#        # Initialising the CNN
#        model = Sequential()
#        # Convolution + Pooling Layer 
#        model.add(Conv2D(64, (5, 5), padding='same', input_shape=input_shape, activation='relu'))
#        model.add(BatchNormalization())
#        model.add(MaxPooling2D(pool_size=(4, 4)))
#        # Convolution + Pooling Layer 
#        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#        model.add(BatchNormalization())
#        model.add(MaxPooling2D(pool_size=(2, 2)))
#        # Convolution + Pooling Layer 
#        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#        model.add(BatchNormalization())
#        model.add(MaxPooling2D(pool_size=(2, 2)))
##        # Convolution + Pooling Layer 
##        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
##        model.add(MaxPooling2D(pool_size=(2, 2)))
#        
#        # Flattening
#        model.add(Flatten())
#        # Fully connection
#        model.add(Dense(64, activation='relu'))
#        model.add(Dropout(p))
##        model.add(Dense(64, activation='relu'))
##        model.add(Dense(64, activation='relu'))
##        model.add(Dropout(p/2))
#        model.add(Dense(1, kernel_initializer='normal',activation='sigmoid'))
#        
#        # Compiling the CNN
#        optimizer = Adam(lr=1e-3)
#        metrics=['accuracy']
#        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
#        return model
#    
#    
#    
#def create_model2(p,input_shape,nb_classes):
#    model = Sequential()
#
#    model.add(Conv2D(32, (5, 5), padding='same', input_shape=input_shape, activation='relu'))
#    model.add(MaxPooling2D(pool_size=(4,4)))
#    model.add(Conv2D(32, 5, 5, activation='relu'))
#    model.add(MaxPooling2D(pool_size=(2,2)))
#    model.add(Dropout(0.25))
#
#    model.add(Flatten())
#    model.add(Dense(128, activation='relu'))
#    model.add(Dropout(0.5))
#    model.add(Dense(nb_classes, activation='softmax'))
#    
#    model.compile(loss='binary_crossentropy',
#              optimizer='adam',
#              metrics=['accuracy'])
#    return model




#train_datagen = ImageDataGenerator(shear_range=0.2,
#                                   zoom_range=0.2,
#                                   rotation_range=20,
#                                   width_shift_range=0.2,
#                                   height_shift_range=0.2,
#                                   horizontal_flip=True,
#                                   rescale = 1./255)

# https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
     
train_datagen = ImageDataGenerator(rescale = 1./255);

train_set = train_datagen.flow_from_directory(path_train,
                                              target_size = (img_width, img_height),
                                              color_mode = 'rgb',
                                              batch_size = batch_size_num ,
                                              class_mode = 'binary',
                                              shuffle = False,
                                              seed = 2019)

test_datagen = ImageDataGenerator(rescale = 1./255);

test_set = test_datagen.flow_from_directory(path_test,
                                              target_size = (img_width, img_height),
                                              color_mode = 'rgb',
                                              batch_size = batch_size_num ,
                                              class_mode = 'binary',
                                              shuffle = False,
                                              seed = 2019)
test_set.reset()

model2 = create_model_matlab(p=0.2, input_shape=(img_width, img_height, 3),decay_rate=0.004)  

history = model2.fit_generator(train_set,
                           steps_per_epoch=400/batch_size_num,
                             epochs = 1000,
                             validation_data = test_set,
                             validation_steps = 200/batch_size_num)

#model.save('CNN_SpecPlot_17DatasetTrain_ProbesTest.h5')
#model2.save('CNN_SpecPlot_17DatasetTrain_ProbesTest2.h5')
model2.save('CNN_SpecPlot_17DatasetTrain_ProbesTest3.h5')
#    model = load_model('Train13Dataset_motifsplit.h5',custom_objects={'AttentionLSTM': AttentionLSTM})

#model2 = load_model('CNN_SpecPlot_17DatasetTrain_ProbesTest2.h5',custom_objects={'AttentionLSTM': AttentionLSTM})

#test_set.reset()
#prediction = model2.predict_generator(test_set)
#test_filenames = test_set.filenames
#
#train_predict = model2.predict_generator(train_set)
#train_filenames = train_set.filenames
#
#test_results = pd.DataFrame({'Filename' : test_filenames,
#                             'Pred_Vals': prediction[:,0]})
#    
#train_results = pd.DataFrame({'Filename' : train_filenames,
#                             'Pred_Vals': train_predict[:,0]})
#    
#labels = (train_set.class_indices)  
    
#test_results = pd.DataFrame({

#predicted_class_indices=np.argmax(prediction,axis=1)


#labels = dict((v,k) for k,v in labels.items())
#predictions = [labels[k] for k in predicted_class_indices]
# To see filenames type filenames = test_set.filenames


### Plotting train and test accuracy plots

train_acc = history.history['acc']
validation_acc = history.history['val_acc']
train_loss = history.history['loss']
validation_loss = history.history['val_loss']
epochs = range(0,1000)
plt.plot(epochs,train_acc,'bo',label='Training Accuracy')
plt.plot(epochs,validation_acc,'b',label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.figure()
plt.plot(epochs,train_loss,'bo',label='Training Loss')
plt.plot(epochs,validation_loss,'b',label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

