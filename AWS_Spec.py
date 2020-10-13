from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import boto3 as boto
import os
import matplotlib.pyplot as plt
access_key = os.environ['AKIA4HOYOH4X43CEOJYV']
access_secret_key = os.environ['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']

conn = boto.connect_s3(access_key, access_secret_key)
bucket = conn.get_bucket('aravindsamala')
file_key = bucket.get_key('312_AUDIO.wav')
file_key.get_contents_to_filename('312_AUDIO.wav')
[Fs, x] = audioBasicIO.read_audio_file('312_AUDIO.wav')
F, f_names, time = ShortTermFeatures.spectrogram(x, Fs, 0.050 * Fs, 0.025 * Fs)
print(F.shape)




"""
fstep = int(num_fft / 5.0)
frequency_ticks = range(0, int(num_fft) + fstep, fstep)
frequency_tick_labels = \
    [str(sampling_rate / 2 -
         int((f * sampling_rate) / (2 * num_fft)))
     for f in frequency_ticks]
ax.set_yticks(frequency_ticks)
ax.set_yticklabels(frequency_tick_labels)
t_step = int(count_fr / 3)


time_ticks = range(0, count_fr, t_step)
time_ticks_labels = \
    ['%.2f' % (float(t * step) / sampling_rate) for t in time_ticks]
ax.set_xticks(time_ticks)
ax.set_xticklabels(time_ticks_labels)
ax.set_xlabel('time (secs)')
ax.set_ylabel('freq (Hz)')
imgplot.set_cmap('jet')
plt.colorbar()
plt.show()
"""
