from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import np_utils
from python_speech_features import mfcc, logfbank
from scipy.io import wavfile
import librosa, librosa.display
# from keras.layers import Conv2D, MaxPool2D, Flatten, LSTM
# from keras.layers import Dropout, Dense, TimeDistributed
# from keras.models import Sequential
# from keras.utils import np_utils
import tensorflow as tf
from sklearn.model_selection import train_test_split


from tqdm import tqdm
mypath = '../data/AudioWAV/'
def build_rand_feat():
    X = []
    y =[]
    _min, _max = float('inf'),-float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index, p= prob_dist)
        file = np.random.choice(emotions[emotions.emotion == rand_class].index)
        rate, wav = wavfile.read(mypath+file)
        label = emotions.at[file, 'emotion']
        rand_index = np.random.randint(0,wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample,rate,numcep = config.nfeat, nfilt = config.nfilt, nfft =config.nfft).T
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample)
        y.append(classes.index(label))
    X, y= np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
    y = np_utils.to_categorical(y,num_classes =10 )
    return X,y
class Config:
    def __init__(self, mode = 'conv', nfilt = 26, nfeat = 13, nfft = 512, rate = 16000):
        self.mode = mode
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.rate = rate
        self.step = int(rate/10)

emotions = pd.read_csv('../data/df2.csv')


n_samples = 2*int(emotions['length'].sum() / 0.1 )
classes = list(np.unique(emotions.emotion))
class_dist = emotions.groupby(['emotion'])['length'].mean()


prob_dist = class_dist / class_dist.sum()
choices = np.random.choice(class_dist.index, p = prob_dist)
emotions.set_index('filename',inplace = True)

config = Config(mode = 'conv')


