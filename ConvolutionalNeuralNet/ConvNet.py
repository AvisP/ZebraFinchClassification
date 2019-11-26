# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 14:58:11 2019

@author: Avishek Paul
"""

import os
from os import walk
import tensorflow as tf
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization

#from keras.datasets import mnist
#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#def _parse_function(filename, label):
#    image_string = tf.read_file(filename)
#    image_decoded = tf.image.decode_jpeg(image_string, channels=1)
#    image = tf.cast(image_decoded, tf.float32)
##    image = tf.image.resize_images(image, [64, 64])
#    return image, label
#
#def train_preprocess(image, label):
#    image = tf.image.random_flip_left_right(image)
#
#    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
#    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
#
#    # Make sure the image is still in [0, 1]
#    image = tf.clip_by_value(image, 0.0, 1.0)
#
#    return image, label

img_width = 128;
img_height = 128;
batch_size_num = 32;

path_train = "F:\data_for_avishek\LSTMGeneric2\RP_Images\Train\\"
path_test = "F:\data_for_avishek\LSTMGeneric2\RP_Images\Test\\"

def create_model(p, input_shape):
        # Initialising the CNN
        model = Sequential()
        # Convolution + Pooling Layer 
        model.add(Conv2D(64, (5, 5), padding='same', input_shape=input_shape, activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(4, 4)))
        # Convolution + Pooling Layer 
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Convolution + Pooling Layer 
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
#        # Convolution + Pooling Layer 
#        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        # Flattening
        model.add(Flatten())
        # Fully connection
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(p))
#        model.add(Dense(64, activation='relu'))
#        model.add(Dense(64, activation='relu'))
#        model.add(Dropout(p/2))
        model.add(Dense(1, kernel_initializer='normal',activation='sigmoid'))
        
        # Compiling the CNN
        optimizer = Adam(lr=1e-3)
        metrics=['accuracy']
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
        return model
    
def create_model2(p,input_shape,nb_classes):
    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(4,4)))
    model.add(Conv2D(32, 5, 5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model

train_datagen = ImageDataGenerator(rescale = 1./255)

train_set = train_datagen.flow_from_directory(path_train,
                                              target_size = (img_width, img_height),
                                              color_mode = 'grayscale',
                                              batch_size = batch_size_num ,
                                              class_mode = 'binary',
                                              shuffle = False,
                                              seed = 2019)

test_datagen = ImageDataGenerator(rescale = 1./255);

test_set = test_datagen.flow_from_directory(path_test,
                                              target_size = (img_width, img_height),
                                              color_mode = 'grayscale',
                                              batch_size = batch_size_num ,
                                              class_mode = 'binary',
                                              shuffle = False,
                                              seed = 2019)


model = create_model(p=0.6, input_shape=(img_width, img_height, 1))  

model.fit_generator(train_set,
                           steps_per_epoch=8000/batch_size_num,
                             epochs = 500,
                             validation_data = test_set,
                             validation_steps = 2000/batch_size_num)

model.save('CNN_RecPlot1.h5')
#    model = load_model('Train13Dataset_motifsplit.h5',custom_objects={'AttentionLSTM': AttentionLSTM})

model= create_model2(p=0.6,input_shape=(img_width, img_height, 1),nb_classes=1)

model.fit_generator(train_set,
                           steps_per_epoch=8000/batch_size_num,
                             epochs = 500,
                             validation_data = test_set,
                             validation_steps = 2000/batch_size_num)

# Viewing filenames from flow_from_directory : test_set.filenames
# Viewing classes from flow_from_directory : test_set.classes
# View labels : test_set.class_indices

test_set.reset()

prediction = model.predict_generator(test_set,verbose=1)
predicted_class_indices=np.argmax(prediction,axis=1)

labels = (train_set.class_indices)
labels = dict((v,k) for k,v in labels.items())
prediction_label = [labels[k] for k in predicted_class_indices]

filenames=test_set.filenames

import pandas as pd
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":prediction})
results.to_csv("results.csv",index=False)

y_true = test_set.classes

from sklearn.metrics import confusion_matrix

y_pred = prediction> 0.5
confusion_matrix(y_true, y_pred)



# Indivdiual plot and label

#from keras.preprocessing import image
#
#f_test = []
#file_list = []
#for (dirpath, dirnames, filenames) in walk(path_test):
#    for file in filenames:
#        f_test.append(os.path.join(dirpath,file))
#        file_list.append(os.path.splitext(file)[0])
##        
##        
##labels = [];        
##for name in filenames:
##        
##    if name.find('undir')<0:
##        labels.append('Directed')
##    else:
##        labels.append('Undirected')
##        
#img = image.load_img(f_test[0],color_mode='grayscale',target_size = (img_width, img_height));
#x = image.img_to_array(img);
#x = np.asarray(x, dtype=K.floatx())
#test_image = np.expand_dims(x,axis=0)
#x_test = test_image.reshape(test_image.shape[0],img_width,img_height,1)
#result = model.predict(x_test)
#result_class = model.predict_classes(x_test)
#
#x_plt = np.reshape(x,(128,128));
#plt.imshow(x_plt,cmap='gray');
#
#predict_label = model.predict( )

    
