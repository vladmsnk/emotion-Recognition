from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
from scipy.io import wavfile

class Extract:
    def __init__(self, path):
        self.path = path

    def extract(self):
        filenames = [f for f in listdir(self.path) if isfile(join(self.path, f))]
        emotion = pd.DataFrame(filenames).loc[:, 0].apply(lambda x: x.split('_')[2])
        filnm = pd.Series(filenames)
        maindata = pd.concat({"filename": filnm, "emotion": emotion}, axis=1)
        maindata.set_index('filename', inplace=True)
        return maindata

    def _makedata(self):
         self.data = self.extract()

    def _add_len(self):
        for f in self.data.index:
            rate, signal = wavfile.read(self.path+f)
            self.data.at[f,'length'] = signal.shape[0] / rate #signal per second

    def _add_info(self):
        self.classes = list(np.unique(self.data.emotion))
        self.class_distr = self.data.groupby(['emotion'])['length'].mean()


    def save_data(self):
        self.data.to_csv('../data/maindata.csv')

    def __call__(self):
        self.extract()
        self.makedata()
        self._add_len()
        self.add_info()
        self.save_data()
