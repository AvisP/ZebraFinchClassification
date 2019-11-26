
Song_extract_Generic2.py - Script for reading all the song files in individual dataset and split them into overlapping segments, generate labels based on the name of the song files and store in a format so that LSTM network can operate on it. The LSTM network used is as mentioned in repository https://github.com/titu1994/LSTM-FCN

Visualization_Generic2.py - Once the training is complete it takes the prediction scores for each segment and puts the segments together to show how the accuracy of the song performed. Example provided are

![Directed_Song_Visualization](blk12_dir1_new.png)
