from os import listdir
from os.path import isfile, join, exists
import pandas as pd
import numpy as np
from scipy.io import wavfile
from configs.config import Config


class Extract():
    def __extract(self):
        filenames = [f for f in listdir(Config.audpath) if isfile(join(Config.audpath, f))]
        emotion = pd.DataFrame(filenames).loc[:, 0].apply(lambda x: x.split('_')[2])
        filnm = pd.Series(filenames)
        maindata = pd.concat({"filename": filnm, "emotion": emotion}, axis=1)
        maindata.set_index('filename', inplace=True)
        self.data = maindata
    def get_data(self):
        if exists(Config.savedatapath):
            return pd.read_csv(Config.savedatapath)
        else:
            print("maindata.csv was not created!")

    def __add_len(self):
        for f in self.data.index:
            rate, signal = wavfile.read(Config.audpath+f)
            self.data.at[f,'length'] = signal.shape[0] / rate  #signal per second

    def __add_info(self):
        self.classes = list(np.unique(self.data.emotion))
        self.class_distr = self.data.groupby(['emotion'])['length'].mean()


    def __save_data(self):
        self.data.to_csv(Config.savedatapath)

    def __change1(self):
        self.data.loc[(self.data['emotion'] == 'DIS'), 'emotion'] = 'ANG'
        self.data.loc[(self.data['emotion'] == 'FEA'), 'emotion'] = 'SAD'

    def __call__(self):
        self.__extract()
        self.__add_len()
        self.__add_info()
        self.__change1()
        self.__save_data()
