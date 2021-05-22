import os
class Config:
    nfilt = 26
    nfeat = 13
    nfft = 512
    ratec = 16000
    step = int(ratec /10)
    n_epoch = 10
    batch_size = 32
    model_path = '../model/mod.model'
    p_path1 = '../pickle/pickle1.p'
    p_path2 = '../pickle/pickle2.p'
    audpath = '../data/AudioWAV/'
    savedatapath = '../data/maindata.csv'


