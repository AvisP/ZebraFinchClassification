# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:55:23 2019

@author: AvishekPaul
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import os
from os import walk


""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)    

""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):        
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

""" plot spectrogram"""
def plotstft(audiopath,folder,subfolder,filename, binsize=2**10, plotpath=None, colormap="jet"):
#    samples, samplerate = librosa.load(audiopath)
    samplerate, samples = wav.read(audiopath)

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel

    timebins, freqbins = np.shape(ims)

    print("timebins: ", timebins)
    print("freqbins: ", freqbins)
    
    plt.figure(frameon=False)
    fig,ax = plt.subplots(1)

    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
#    plt.colorbar()

#    plt.xlabel("time (s)")
#    plt.ylabel("frequency (hz)")
    plt.xlim([0, timebins-1])
    plt.ylim([0, freqbins])
    
    plt.axis('off');
    ax.get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
    ax.get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis
    fig.set_size_inches(2, 2) 
    
    plt.savefig("{}/{}.png".format(folder+subfolder,filename), bbox_inches='tight',pad_inches=0,dpi=128)
#    xlocs = np.float32(np.linspace(0, timebins-1, 5))
#    plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
#    ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
#    plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])

    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()

    plt.clf()

    return ims

#ims = plotstft(filepath)
#folder = 'F:\data_for_avishek\LSTMGeneric2\RP_Images\Test\blk12\\';
#filename =  'motif.10775.5782blk12_undir3_new.wav';
#savefolder = folder = 'F:\data_for_avishek\LSTMGeneric2\Spectral_Images\Train\\' 
    
#######################   Training Image Generation Section  ################
#path_train = "F:\data_for_avishek\LSTMGeneric2\Train\\"
path_train = "F:\data_for_avishek\LSTMGeneric3\Train\\"

f_train = []
file_list_train = []
for (dirpath, dirnames, filenames) in walk(path_train):
    for file in filenames:
        f_train.append(os.path.join(dirpath,file))
        file_list_train.append(os.path.splitext(file)[0])
        
data_out_train = []

for path in f_train:
    
    if path.find('undir')<0:
        data_out_train.append('Directed')
    else:
        data_out_train.append('Undirected')
        
#savefolder_Train = 'F:\data_for_avishek\LSTMGeneric2\Spectral_Images\Train\\'         
savefolder_Train = 'F:\data_for_avishek\LSTMGeneric3\Spectral_Images\Train\\'

for idx,val in enumerate(data_out_train):
    print(idx);
    ims = plotstft(f_train[idx],savefolder_Train,data_out_train[idx],file_list_train[idx],binsize=2**8)
    
###################### Testing Image Generation Section ######################


#path_test = "F:\data_for_avishek\LSTMGeneric2\Test\\"
path_test = "F:\data_for_avishek\LSTMGeneric3\Test\\"

f_test = []
file_list_test = []

for (dirpath, dirnames, filenames) in walk(path_test):
    for file in filenames:
        f_test.append(os.path.join(dirpath,file))
        file_list_test.append(os.path.splitext(file)[0])
        
data_out_test = []

for path in f_test:
    
    if path.find('undir')<0:
        data_out_test.append('Directed')
    else:
        data_out_test.append('Undirected')
        
#savefolder_Test = 'F:\data_for_avishek\LSTMGeneric2\Spectral_Images\Test\\'
savefolder_Test = 'F:\data_for_avishek\LSTMGeneric3\Spectral_Images\Test\\'

for idx,val in enumerate(data_out_test):
    print(idx);
    ims = plotstft(f_test[idx],savefolder_Test,data_out_test[idx],file_list_test[idx],binsize=2**8)    
    