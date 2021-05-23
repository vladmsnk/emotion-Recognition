import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from configs.config import Config
from python_speech_features import mfcc, logfbank
from keras.utils import np_utils
import os
import pickle


class Preprocess():

    def __load_data(self):
        self.df = pd.read_csv(Config.savedatapath)
        self.df.set_index('filename',inplace =True)

    def __info(self):
        self.n_samples = 2*int(self.df['length'].sum()/ 0.1)
        self.class_dist = self.df.groupby(['emotion'])['length'].mean()
        self.prob_dist = self.class_dist / self.class_dist.sum()
        self.choices = np.random.choice(self.class_dist.index, p = self.prob_dist)
        self.classes = list(np.unique(self.df.emotion))

    def get_info(self):
        return (self.n_samples,self.class_dist,self.prob_dist,self.choices,self.classes)


    def build_rand_feat(self):

        X = []
        y = []
        minimum, maximum = float('inf'), -float('inf')
        for _ in tqdm(range(self.n_samples)):
            rand_class = np.random.choice(self.class_dist.index, p = self.prob_dist)
            file = np.random.choice(self.df[self.df.emotion == rand_class].index)
            rate, wav = wavfile.read(Config.audpath + file)
            emotion = self.df.at[file, 'emotion']
            rand_index = np.random.randint(0, wav.shape[0] - Config.step)
            sample = wav[rand_index: rand_index + Config.step]

            X_sample = mfcc(sample, rate, numcep= Config.nfeat, nfilt = Config.nfilt, nfft = Config.nfft).T
            minimum = min(np.amin(X_sample), minimum)
            maximum = max(np.amax(X_sample), maximum)
            X.append(X_sample)
            y.append(self.classes.index(emotion))
        X, y = np.array(X), np.array(y)
        X = (X - minimum) / (maximum - minimum)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2],1)
        y = np_utils.to_categorical(y,num_classes =4)
        with open(Config.p_path1, 'wb') as handle1:
            pickle.dump(X, handle1, protocol=2)
        with open(Config.p_path2, 'wb') as handle2:
            pickle.dump(y, handle2, protocol=2)
        return X, y

    def __call__(self):
        self.__load_data()
        self.__info()



