import pandas as pd
import numpy as np



class Preprocess:

    def load_data(self):
        self.df = pd.read_csv('../data/maindata.csv')
    def info(self):
        self.n_samples = 2*int(self.df['length'].sum()/ 0.1)
        self.class_dist = self.df.groupby(['emotion'])['length'].mean()
        self.prob_dist = self.class_dist / self.class_dist.sum()
        self.choices = np.random.choice(self.class_dist.index, p = self.prob_dist)


