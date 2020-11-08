import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import pandas as pd
import pickle

sample_rate, samples = wavfile.read('/Volumes/Macintosh HD - Data/Users/admin/Documents/HD Drive/DataProjects_Data/DepressionData/303.wav')
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)



'''
train_split = pd.read_csv('train_split.csv')


for id in train_split['Participant_ID']:
    print(id)
'''
'''
a_file = open("Spec.pkl", "rb")
output = pickle.load(a_file)
length_dict = {key: len(value) for key, value in output.items()}
length_key = length_dict['key']  # length of the list stored at `'key'` ...

'''