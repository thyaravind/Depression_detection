from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
from zipfile import ZipFile
import numpy as np
import pandas as pd
import pickle

train_split = pd.read_csv('/Volumes/Macintosh HD - Data/Users/admin/Documents/HD Drive/DataProjects/DepressionData/train_split.csv')


featureDict = {}

def ExtractSpec(id):
    [Fs, x] = audioBasicIO.read_audio_file("/Volumes/Macintosh HD - Data/Users/admin/Documents/HD Drive/DataProjects/DepressionData/audio/{}_AUDIO.wav".format(id))
    F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050 * Fs, 0.025 * Fs)
    return F



for id in train_split['Participant_ID']:
    featureDict[f'{id}'] = ExtractSpec(id)
    print(featureDict[f'{id}'].shape)



with open("/Volumes/Macintosh HD - Data/Users/admin/Documents/HD Drive/DataProjects/DepressionData/feature.pkl", "wb") as file:
    pickle.dump(featureDict,file)