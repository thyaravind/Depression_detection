import os
import librosa
import librosa.display
import IPython.display as ipd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zipfile import ZipFile
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from pydub import AudioSegment


train_split = pd.read_csv('Data/train_split.csv')

FRAME_SIZE = 2048
HOP_SIZE = 512

def ExtractSpectrogram(id,i):
    audio, sampling_rate = librosa.load(f'/Volumes/Macintosh HD - Data/Users/admin/Documents/HD Drive/DataProjects_Data/DepressionData/audio/{id}_AUDIO.wav')
    audio = audio[i:i+10000]
    audio_stft = librosa.stft(audio, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    audio_data = librosa.power_to_db(np.abs(audio_stft) ** 2)
    return audio_data



# %%
    spec_array = np.zeros(shape=(1,1025,20,150))
    id = 305
    j = 0
    for i in range(0,1500000,10000):
        audio_data = ExtractSpectrogram(id,i)
        audio_3d = np.expand_dims(audio_data,axis = 0)
        spec_array[:,:,:,j] = audio_3d
        j+=1


#%%
for id in train_split['Participant_ID']:
