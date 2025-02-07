import kagglehub

# Download latest version
# path = kagglehub.dataset_download("mathurinache/the-lj-speech-dataset")
path = 'C:\Users\Danya\.cache\kagglehub\datasets\mathurinache\the-lj-speech-dataset\versions\1'

print("Path to dataset files:", path)
# Path to dataset files: C:\Users\Danya\.cache\kagglehub\datasets\mathurinache\the-lj-speech-dataset\versions\1
import tensorflow as tf

import os
import numpy as np
from scipy.io import wavfile
import uuid
import pandas as pd

dir = 'C:\\Users\\Danya\\.cache\\kagglehub\\datasets\\mathurinache\\the-lj-speech-dataset\\versions\\1\\LJSpeech-1.1\\wavs'
dir_cut_data = 'cut\\'
Fs = 22050
step = 22050
obj = os.scandir(dir)

for entry in obj :
    if entry.is_file():
        samplerate, data = wavfile.read(entry)
        for i in range(step, len(data), step):
            unique_filename = dir_cut_data + str(uuid.uuid4()) + '.wav'
            wavfile.write(unique_filename, samplerate, data[i-step:i].astype(np.int16))