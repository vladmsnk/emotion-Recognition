import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
from tensorflow import keras
from configs.config import Config
from sklearn.metrics import accuracy_score
class Predict:
    def __init__(self, aud_pred_path):
        self.aud_pred_path = aud_pred_path
        self.model = keras.models.load_model('../model/mod.model')
        self.x = self.__to_mcc()


    def __to_mcc(self):
        rate, wav = wavfile.read(self.aud_pred_path)
        rand_index = np.random.randint(0, wav.shape[0] - Config.step)
        sample = wav[rand_index: rand_index + Config.step]
        x = mfcc(sample, rate, numcep=Config.nfeat, nfilt=26, nfft=Config.nfft).T
        x = np.expand_dims(x, axis=0)
        x = np.expand_dims(x, axis=0).reshape((1, 13, 9, 1))
        return x
    def predict(self):
        self.prediction = self.model.predict(self.x)
        if np.where(self.prediction == 1)[0][0] == 0:
            return 'ANG'
        elif np.where(self.prediction == 1)[0][0] == 1:
            return 'HAP'
        elif np.where(self.prediction == 1)[0][0] == 2:
            return 'NEU'
        else:
            return 'SAD'




