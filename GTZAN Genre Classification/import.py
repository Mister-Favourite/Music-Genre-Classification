import numpy as np
import librosa
import os
import csv

# Source: https://medium.com/@sdoshi579/classification-of-music-into-different-genres-using-keras-82ab5339efe0

header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'

for i in range(1, 21):
    header += f' mfcc{i}'

header += ' label'
header = header.split()

file = open('GTZAN Genre Classification\GTZAN Dataset\data.csv', 'w', newline='')
with file:
    writer = csv.writer(file)
    writer.writerow(header)

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

for g in genres:
    for filename in os.listdir(f'./GTZAN Genre Classification/GTZAN Dataset/{g}'):
        
        songname = f'./GTZAN Genre Classification/GTZAN Dataset/{g}/{filename}'
        y, sr = librosa.load(songname, mono=True, duration=30)

        # Features: 
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)[0]
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {g}'

        file = open('GTZAN Genre Classification\GTZAN Dataset\data.csv', 'a', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(to_append.split())