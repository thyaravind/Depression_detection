from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pickle

train_split = pd.read_csv('train_split.csv')


featureDict = {}

def ExtractSpec(id):
    with ZipFile("/Volumes/Macintosh HD - Data/Users/admin/Downloads/{}_P.zip".format(id), 'r') as zip:
        audio = zip.extract("{}_AUDIO.wav".format(id), 'audio')
        [Fs, x] = audioBasicIO.read_audio_file("audio/{}_AUDIO.wav".format(id))
        F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
        return F



for id in train_split['Participant_ID']:
    SpecDict[f'{id}'] = ExtractSpec(id)
    print(ExtractSpec(id).shape)