# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 16:21:44 2019

@author: Avishek Paul
"""
# Code for generating images from the mp3 files. First 13 datasets conisdered only
# Each dataset split inot 80 - 20

import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
from os import walk
from sklearn.model_selection import train_test_split
from pyts.image import RecurrencePlots
import matplotlib.pyplot as plt

## Parameters
#n_samples, n_features = 100, 144
#
## Toy dataset
#rng = np.random.RandomState(41)
#X = rng.randn(n_samples, n_features)
#
## Recurrence plot transformation
#rp = RecurrencePlots(dimension=1,
#                     epsilon='percentage_points',
#                     percentage=30)
#X_rp = rp.fit_transform(X)
#
## Show the results for the first time series
#plt.figure(figsize=(8, 8))
#plt.imshow(X_rp[0], cmap='binary', origin='lower')
#plt.show()

### Training Data Section ###
path_train = "F:\data_for_avishek\LSTMGeneric2\Train\\"


# Get list of files in directory
f_train = []
file_list = []
for (dirpath, dirnames, filenames) in walk(path_train):
    for file in filenames:
        f_train.append(os.path.join(dirpath,file))
        file_list.append(os.path.splitext(file)[0])
        
total_data_train = []
total_data_time_train = []
data_out_train = []
max_size=0 

for path in f_train:
    
    y, sr = librosa.load(path)
    time = np.arange(0,len(y)/sr,1/sr);
    total_data_train.append(y)
    total_data_time_train.append(time)
    
    if len(y)>max_size:
        max_size = len(y)
    
    if path.find('undir')<0:
        data_out_train.append('Directed')
    else:
        data_out_train.append('Undirected')
        
X_train = np.empty([len(total_data_train),max_size])
rp = RecurrencePlots(dimension=1,
                     epsilon='percentage_points',
                     percentage=30)

folder = 'F:\data_for_avishek\LSTMGeneric2\RP_Images2\Train\\' 
#folderD = 'F:\data_for_avishek\LSTMGeneric2\RP_Images2\Train\Directed\\'  
#folderU = r'F:\data_for_avishek\LSTMGeneric2\RP_Images2\Train\Undirected\\'  


#num = 78;

#fig = plt.figure(frameon=False)
#ax = plt.gca()
#im = ax.imshow(data_list[0],...)
#X_train[idx,:len(val)] = val
#X_rp = rp.fit_transform(val.reshape(1,len(val)))

for idx,val in enumerate(total_data_train):
    if idx<10:
        print(idx)
#        X_train[idx,:len(val)] = val
        X_rp = rp.fit_transform(val.reshape(1,len(val)))
        plt.figure(frameon=False)
        fig,ax = plt.subplots(1)
        plt.imshow(X_rp[0], cmap='binary', origin='lower')
        plt.axis('off');
        ax.get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
        ax.get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis
        fig.set_size_inches(2, 2)
        plt.savefig("{}/{}.png".format(folder+data_out_train[idx],file_list[idx]), bbox_inches='tight',pad_inches=0,dpi=128)
#        if data_out_train[idx]=='Directed':
#            plt.savefig("{}/{}.png".format(folderD,file_list[idx]), bbox_inches='tight',pad_inches=0,dpi=128)
#        else:
#            plt.savefig("{}/{}.png".format(folderU,file_list[idx]), bbox_inches='tight',pad_inches=0,dpi=128)
        plt.clf()
        del X_rp
  
  
### Testing Data Section ###
        ## Note : RESTART KERNL ##
path_test = "F:\data_for_avishek\LSTMGeneric2\Test\\"


# Get list of files in directory
f_test = []
file_list_test = []
for (dirpath, dirnames, filenames) in walk(path_test):
    for file in filenames:
        f_test.append(os.path.join(dirpath,file))
        file_list_test.append(os.path.splitext(file)[0])
        
total_data_test = []
total_data_time_test = []
data_out_test = []

max_size=0 
for path in f_test:
    
    y, sr = librosa.load(path)
    time = np.arange(0,len(y)/sr,1/sr);
    total_data_test.append(y)
    total_data_time_test.append(time)
    
    if len(y)>max_size:
        max_size = len(y)
    
    if path.find('undir')<0:
        data_out_test.append('Directed')
    else:
        data_out_test.append('Undirected')
        
X_test = np.empty([len(total_data_test),max_size])
rp = RecurrencePlots(dimension=1,
                     epsilon='percentage_points',
                     percentage=30)

folder = 'F:\data_for_avishek\LSTMGeneric2\RP_Images\Test\\' 

for idx,val in enumerate(total_data_test):
    print(idx)
    X_rp = rp.fit_transform(val.reshape(1,len(val)))
    plt.figure(frameon=False)
    fig,ax = plt.subplots(1)
    plt.imshow(X_rp[0], cmap='binary', origin='lower')
    plt.axis('off');
    ax.get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
    ax.get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis
    fig.set_size_inches(2, 2) 
    plt.savefig("{}/{}.png".format(folder+data_out_test[idx],file_list_test[idx]), bbox_inches='tight',pad_inches=0,dpi=128)
    plt.clf()
    del X_rp 

## Recurrence plot transformation
#rp = RecurrencePlots(dimension=1,
#                     epsilon='percentage_points',
#                     percentage=30)
#X_rp = rp.fit(X_train)    
#
#X_rp = rp.fit_transform(val.reshape(1,len(val)))
#
#plt.figure(figsize=(8, 8))
#plt.imshow(X_rp[0], cmap='binary', origin='lower')
#plt.show()  
#
#
#plt.figure(idx, frameon=False)
#fig,ax = plt.subplots(1)
#plt.imshow(X_rp[0], cmap='binary', origin='lower')
#plt.axis('off');
#ax.get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
#ax.get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis
#fig.set_size_inches(2, 2) 
#plt.savefig("{}/{}.png".format(folder,file_list[idx]), bbox_inches='tight',pad_inches=0,dpi=128)

y_train = np.empty([1,1])