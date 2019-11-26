# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 12:10:52 2018

@author: SakataWoolley
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix

def song_split_accuracy(song_split,time,split_win,prediction,filename):
#    import math
#    cm = plt.get_cmap('cool')
    split_num = len(song_split);
    first_song_idx = int(split_num/2);
    accuracy = 0;
    
    if filename.find('undir')<0:
        pred_indx = 1;  
    else:
        pred_indx=1;    
    
#    fig = plt.gcf()
#    ax1 = plt.subplot(211)
    for i in range(first_song_idx-1):
#        plt.plot(time[i*split_win:(i+1)*split_win],song_split[i,:],color=cm(math.floor(prediction[i,0]*255)))
        accuracy = accuracy + prediction[i,pred_indx];
#        if prediction[i,0]>0.5 and filename.find('undir')<0:
#            ax1.axvspan(time[i*split_win],time[(i+1)*split_win], facecolor='0.5', alpha=0.25, label='temp')
#        elif prediction[i,0]<0.5 and filename.find('undir')>1:
#            ax1.axvspan(time[i*split_win],time[(i+1)*split_win], facecolor='0.5', alpha=0.25, label='temp')
            
#    plt.plot(time[(i+1)*split_win:],song_split[(i+1),:(len(time)-(i+1)*split_win)],color=cm(math.floor(prediction[(i+1),0]*255)))
#    if prediction[i,0]>0.5 and filename.find('undir')<0:
#        ax1.axvspan(time[i*split_win],time[(i+1)*split_win], facecolor='0.5', alpha=0.25, label='temp')
#    elif prediction[i,0]<0.5 and filename.find('undir')>1:
#        ax1.axvspan(time[i*split_win],time[(i+1)*split_win], facecolor='0.5', alpha=0.25, label='temp')
#    plt.title(filename)
    accuracy = accuracy + prediction[(i+1),pred_indx];
    
    accuracy = accuracy/(first_song_idx);
#    alignment = {'horizontalalignment': 'center', 'verticalalignment': 'baseline'}
#    plt.text(1.0, 1.0, accuracy, fontsize=12,**alignment)
    
    accuracy1 = 0;
    
#    ax2 = plt.subplot(212)
#    plt.plot(time[0:int(split_win/2)],song_split[(i+2),:int(split_win/2)],color=cm(math.floor(prediction[(i+2),0]*255)))
    accuracy1 = accuracy1 + prediction[(i+2),pred_indx]
    
    start_idx = i+3;
    for j in range(start_idx,split_num-1):
#        plt.plot(time[(j-start_idx)*split_win+int(split_win/2):(j-start_idx+1)*split_win+int(split_win/2)],song_split[j,:],color=cm(math.floor(prediction[j,0]*255)))
        accuracy1 = accuracy1 + prediction[j,pred_indx]
#        if prediction[j,0]>0.5 and filename.find('undir')<0:
#            ax2.axvspan(time[(j-start_idx)*split_win+int(split_win/2)],time[(j-start_idx+1)*split_win+int(split_win/2)], facecolor='0.5', alpha=0.25, label='temp')
#        elif prediction[j,0]<0.5 and filename.find('undir')>1:
#            ax2.axvspan(time[(j-start_idx)*split_win+int(split_win/2)],time[(j-start_idx+1)*split_win+int(split_win/2)], facecolor='0.5', alpha=0.25, label='temp')
    
#    plt.plot(time[(j+1-start_idx)*split_win+int(split_win/2):],song_split[(j+1),:len(time[(j+1-start_idx)*split_win+int(split_win/2):])],color=cm(math.floor(prediction[(j+1),0]*255)))
#    if prediction[j,0]>0.5 and filename.find('undir')<0:
#        ax2.axvspan(time[(j-start_idx)*split_win+int(split_win/2)],time[(j-start_idx+1)*split_win+int(split_win/2)], facecolor='0.5', alpha=0.25, label='temp')
#    elif prediction[j,0]<0.5 and filename.find('undir')>1:
#        ax2.axvspan(time[(j-start_idx)*split_win+int(split_win/2)],time[(j-start_idx+1)*split_win+int(split_win/2)], facecolor='0.5', alpha=0.25, label='temp')
    accuracy1 = accuracy1 + prediction[(j+1),pred_indx]
    
    accuracy1 = accuracy1/(split_num-start_idx+2);
    res_accuracy = (accuracy+accuracy1)/2;
    return res_accuracy
#    plt.text(1.0, 1.0, accuracy, fontsize=12,**alignment)
#
#    plt.xlabel('Time(s)')
#    fig.set_size_inches(18.5, 10.5)
#    fig.savefig(filename[:-4]+'.png',dpi=1000)
#    plt.close(fig)
#    fig.savefig(filename[:-4]+'.eps',format='eps',dpi=1000)

split_win = 1000;    
 
with open("AllDataset_data2_test.txt", "rb") as fp:   # Unpickling
    data = pickle.load(fp)
    
total_data = data[0];
total_data_time = data[1];
data_out = data[2];
filelist = data[3];

prediction = np.loadtxt('Train13Dataset_motifsplit_pred2.txt', dtype='float32')

cur_song = total_data[0]
time = total_data[0]

ypred_split=[]
start_idx = 0;
end_idx = 0;

for data in total_data:
    end_idx = start_idx + len(data);
    ypred_split.append(prediction[start_idx:end_idx,:])
    start_idx = end_idx;

data_temp = data_out[0:4]

#for i,yout in enumerate(data_temp):#(data_out[4:5]):    
#    print(i)
#    song_split_plot(total_data[i],total_data_time[i],split_win,ypred_split[i],filelist[i])
motif_pred = np.empty((1,1),dtype=float)    
for i in range(len(ypred_split)):
     motif_pred = np.vstack((motif_pred,song_split_accuracy(total_data[i],total_data_time[i],1000,ypred_split[i],filelist[i])))

motif_pred = motif_pred[1:];

# Note : if pred_indx is set as 1 and 0 for the classes then use this loop and motif_pred indicates that the 
# confidence in prediction. No need for making further classification. A value less than 0.5 will indicate an
# incorrect classfication
incorrect_classification=np.where(motif_pred<0.5);
(len(incorrect_classification[0])*100)/len(motif_pred)


# NOTE : if pred_indx is set as 1 for both directed and undirected in the function then motif_pred is giving the 
# confidence score for both the classes and use this loop for classification. It uses 0.5 as default threshold

motif_pred_label = [];
for val in motif_pred:
    if val<0.5:
        motif_pred_label.append('Undirected')
    else:
        motif_pred_label.append('Directed')
        
cm = confusion_matrix(data_out,motif_pred_label)

from plot_confusion_matrix import pretty_plot_confusion_matrix
pretty_plot_confusion_matrix(cm)

from plot_confusion_matrix import plot_confusion_matrix_from_data
plot_confusion_matrix_from_data(data_out,motif_pred_label,columns=['Directed','Undirected'],figsize=[5,5])
#
#i=0;

#for i in range(len(x)-1):
#	plt.plot(x[i:i+2],y[i:i+2],cm(i))    

