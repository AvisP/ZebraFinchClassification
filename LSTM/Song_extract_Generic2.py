# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 11:56:34 2018

@author: SakataWoolley
"""

import librosa
import math
import numpy as np
import os
from os import walk
from sklearn.model_selection import train_test_split


def data_split(song,split_win):
    split_num = math.floor(len(song)/split_win)
    song_split = np.split(song[0:split_num*split_win],split_num);
    
    last_seg = song[split_num*split_win:];
    pad_length = split_win - len(last_seg);
    last_seg = np.hstack([last_seg, np.zeros([ pad_length])]);
    song_split = np.vstack((song_split,last_seg));
    
    split_offset = int(split_win/2);
    split_num = math.floor(len(song[split_offset:])/split_win)
    initial_seg = song[0:split_offset];
    initial_seg = np.hstack([initial_seg, np.zeros([ split_offset])]);
    song_split = np.vstack((song_split,initial_seg));
    
    song_split_offset = np.split(song[split_offset:split_offset+split_num*split_win],split_num)
    song_split = np.vstack((song_split,song_split_offset));
    
    last_seg = song[split_offset+split_num*split_win:];
    pad_length = split_win - len(last_seg);
    last_seg = np.hstack([last_seg, np.zeros([ pad_length])]);
    song_split = np.vstack((song_split,initial_seg));
    
    return song_split

   
path_train = "F:\data_for_avishek\LSTMGeneric2\Train\\"
#path_test = "F:\data_for_avishek\LSTMGeneric2\Test\\"

# Get list of files in directory
f_train = []
for (dirpath, dirnames, filenames) in walk(path_train):
    for file in filenames:
        f_train.append(os.path.join(dirpath,file))

#with open('motif_batch.txt', 'r') as f:
#    x = f.readlines()
#    
#filelist = []
#for line in x:
#    filelist.append(line.rstrip('\n'))    
    
    
#parent_dir = "F:\data_for_avishek\LSTM blk12\\"
#parent_dir = "F:\data_for_avishek\LTSM blk12\\"
#savefilename = "blk12.out"
#savefilename_total = "blk12_data.txt"
    
parent_dir_train = "F:\data_for_avishek\LSTMGeneric2\\"
savefilename_train = "AllDataset_TRAIN2.out"
savefilename_total_train = "AllDataset_data2_train.txt"
#parent_dir_test = "F:\data_for_avishek\LSTMGeneric2\\"
#savefilename_test = "AllDataset_TEST2.out"    

split_win = 1000;

total_data_train = []
total_data_time_train = []
data_out_train = []

for path in f_train:
    
    y, sr = librosa.load(path)
    time = np.arange(0,len(y)/sr,1/sr);
    total_data_train.append(data_split(y,split_win))
    total_data_time_train.append(time)
    
    if path.find('undir')<0:
        data_out_train.append('Directed')
    else:
        data_out_train.append('Undirected')
        
X_train = np.empty([1,1000])
y_train = np.empty([1,1])
for iter,data in enumerate(total_data_train):
    X_train = np.vstack((X_train,data))
    if data_out_train[iter].find('Undirected')<0:
        y_train = np.vstack((y_train,np.ones([len(data),1])))  # Directed is 1
    else:
        y_train = np.vstack((y_train,np.zeros([len(data),1])))  # Undirected is 0
    
X_train = X_train[1:];
y_train = y_train[1:];

#X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.2,random_state=2019)
#savedata = np.hstack((y_train,X_train));
np.savetxt(savefilename_train,np.hstack((y_train,X_train)),delimiter=',')

#np.savetxt(savefilename_test,np.hstack((y_test,X_test)),delimiter=',')
#np.savetxt('AllDataset_total',np.hstack((y_total,X_total)),delimiter=',')

aggregated_total_train = []
aggregated_total_train.append(total_data_train)
aggregated_total_train.append(total_data_time_train)
aggregated_total_train.append(data_out_train)
#aggregated_total_train.append(filelist)

import pickle

with open(savefilename_total_train,"wb") as fp:
    pickle.dump(aggregated_total_train ,fp)
