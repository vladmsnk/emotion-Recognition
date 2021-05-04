import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from configs.config import Config



class Preprocess(Config):
    def load_data(self):
        self.df = pd.read_csv(self.savedatapath)
    def info(self):
        self.n_samples = 2*int(self.df['length'].sum()/ 0.1)
        self.class_dist = self.df.groupby(['emotion'])['length'].mean()
        self.prob_dist = self.class_dist / self.class_dist.sum()
        self.choices = np.random.choice(self.class_dist.index, p = self.prob_dist)
    def build_rand_feat(self):
        X = []
        y = []
        minimum, maximum = float('inf'), -float('inf')
        for _ in tqdm(range(self.n_samples)):
            rand_class = np.random.choice(self.class_dist.index, p = self.prob_dist)
            file = np.random.choice(self.df[self.df.emotion == rand_class].index)
            rate, wav = wavfile.read(self.audpath + file)
            emotion = self.df.at[file, 'emotion']
            rand_index = np.random.randint(0, wav.shape[0] - self.step)

