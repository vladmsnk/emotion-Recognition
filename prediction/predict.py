import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import load_model
import pandas as pd
from tensorflow import keras

from sklearn.metrics import accuracy_score

# def build_prediction(audio_dir):
#     y_true = []
#     y_pred = []
#     fn_prob = {}
#     print("")


data = pd.read_csv('../data/maindata.csv')


model = keras.models.load_model('../model/mod.model')


loss, acc = model.evaluate()