
Song_extract_Generic2.py - Script for reading all the song files in individual dataset and split them into overlapping segments, generate labels based on the name of the song files and store in a format so that LSTM network can operate on it. The LSTM network used is as mentioned in repository https://github.com/titu1994/LSTM-FCN

Visualization_Generic2.py - Once the training is complete it takes the prediction scores for each segment and puts the segments together to show how the accuracy of the song performed. Directed songs are in cyan and Undirected songs in magenta and based on the confidence score of predcition of each section the color will vary. If an incorrect classification is made then background of the section is marked in grey.

**Directed SONG**

![Directed_Song_Visualization](blk12_dir1_new.png)

**Undirected Song**

![Directed_Song_Visualization](blk12_undir1_new.png)
