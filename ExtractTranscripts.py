from zipfile import ZipFile
import numpy as np
import pandas as pd
import pickle

'''
def Extract(id):
    with ZipFile("/Volumes/Macintosh HD - Data/Users/admin/Downloads/{}_P.zip".format(id), 'r') as zip:
        zip.extract("{}_TRANSCRIPT.csv".format(id), 'text')
        

'''
df = pd.read_csv('text/{}_TRANSCRIPT.csv'.format(301))
df
