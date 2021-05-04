from os import listdir
from os.path import isfile, join
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from python_speech_features import mfcc, logfbank
from scipy.io import wavfile
import librosa, librosa.display
from sklearn.model_selection import train_test_split

def calculate_fft(y,rate):
    n = len(y)
    freq = np.fft.rfftfreq(n,d = 1/rate)
    Y = abs(np.fft.rfft(y)/n)
    return (Y,freq)

# def plt_mfccs(mfccs):
#     fig,ax = plt.subplots(nrows = 2,ncols =  3 , sharex = False,sharey = False,figsize = (15,5))
#     fig.suptitle("Mel Frequency Cepstrum Coefficients", size = 15)
#     i = 0
#     for x in range(2):
#         for y in range(3):
#             ax[x,y].set_title(list(mfccs.keys())[i])
#             ax[x,y].imshow(list(mfccs.values())[i], cmap='hot',interpolation = 'nearest')
#             ax[x,y].get_xaxis().set_visible(False)
#             ax[x,y].get_yaxis().set_visible(False)
#             i+=1
mypath = '../data/AudioWAV/'


filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]

number = pd.DataFrame(filenames).loc[:,0].apply(lambda x: x.split('_')[0])

second = pd.DataFrame(filenames).loc[:,0].apply(lambda x: x.split('_')[1])

emotion = pd.DataFrame(filenames).loc[:,0].apply(lambda x: x.split('_')[2])

gender = pd.DataFrame(filenames).loc[:,0].apply(lambda x: x.split('_')[3])

data = {"number": number,
        "second": second,
        "emotion": emotion,
        "gender":gender}
df1 = pd.concat(data, axis = 1) # df1 contain columns of splited filename

filnm = pd.Series(filenames)
for_train = {"filename": filnm,"emotion":emotion}
df2 = pd.concat(for_train,axis =1) #first column - filename second - emotion

df2.set_index('filename',inplace = True)

for f in df2.index:
    rate, signal = wavfile.read(mypath+f)
    df2.at[f,'length'] = signal.shape[0]/rate   #signal per second


classes = list(np.unique(df2.emotion))
class_distr = df2.groupby(['emotion'])['length'].mean()

fig, ax = plt.subplots()
ax.set_title('Class Distribution')
ax.pie(class_distr, labels= class_distr.index, autopct='%1.1f%%',
       shadow = False,startangle = 90)
ax.axis('equal')
#plt.show()

df2.reset_index(inplace=True)

signals = {}
fft = {}
fbank = {}
mfccs = {}
for c in classes:
    wav_file = df2[df2.emotion == c].iloc[0,0]
    signal, rate = librosa.load(mypath + wav_file, sr = 44100) #time series
    signals[c] = signal
    fft[c] = calculate_fft(signal,rate) #fast fourie transform
    bank = logfbank(signal[:rate],rate,nfilt=26,nfft =  1103).T
    fbank[c] = bank
    mel = mfcc(signal[:rate],rate,numcep = 13, nfilt = 26, nfft = 1103).T
    mfccs[c] = mel
df2.to_csv("../data/df2.csv")
