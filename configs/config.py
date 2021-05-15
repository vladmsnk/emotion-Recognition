
class Config:
    def __init__(self, nfilt = 26, nfeat = 13, nfft = 512 , ratec = 16000):
        self.nfilt = nfilt
        self.nfeat = nfeat
        self.nfft = nfft
        self.ratec = ratec
        self.step = int(ratec /10)
        self.audpath = 'data/AudioWAV/'
        self.savedatapath = 'data/maindata.csv'
        self.n_epoch = 75
        self.batch_size = 32




