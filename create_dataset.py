import os
import numpy as np
import soundfile as sf
import librosa
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

def load_and_preprocess_audio(file_path, n_mfcc=20, max_frames=500):
    audio, sr = sf.read(file_path)
    
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    
    if mfcc.shape[1] < max_frames:
        pad_width = ((0, 0), (0, max_frames - mfcc.shape[1]))
        mfcc = np.pad(mfcc, pad_width, mode='constant')
    else:
        mfcc = mfcc[:, :max_frames]
    
    return mfcc.T

def create_tf_dataset(noisy_dir, csv_path, output_dir, n_mfcc=20, max_frames=500):
    df = pd.read_csv(csv_path)
    
    mfcc_features = []
    stoi_labels = []
    filenames = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing files"):
        if row['stoi_score'] == 'error':
            continue
            
        try:
            file_path = os.path.join(noisy_dir, row['filename'])
            mfcc = load_and_preprocess_audio(file_path, n_mfcc, max_frames)
            
            mfcc_features.append(mfcc)
            stoi_labels.append(float(row['stoi_score']))
            filenames.append(row['filename'])
        except Exception as e:
            print(f"Error processing {row['filename']}: {str(e)}")
            continue
    
    mfcc_features = np.array(mfcc_features)
    stoi_labels = np.array(stoi_labels)
    filenames = np.array(filenames)
    
    dataset = tf.data.Dataset.from_tensor_slices({
        'mfcc': mfcc_features,
        'stoi': stoi_labels,
        'filename': filenames
    })
    
    os.makedirs(output_dir, exist_ok=True)
    
    dataset.save(output_dir)
    print(f"\nTF Dataset successfully saved to {output_dir}")
    
    return dataset

if __name__ == "__main__":
    noisy_dir = "/home/danya/develop/datasets/CMU-MOSEI/Audio/vad_noise/"
    csv_path = "/home/danya/develop/datasets/CMU-MOSEI/Audio/stoi_results.csv"
    output_dir = "/home/danya/develop/datasets/CMU-MOSEI/Audio/tf_dataset/"
    
    N_MFCC = 20
    MAX_FRAMES = 500
    
    dataset = create_tf_dataset(noisy_dir, csv_path, output_dir, N_MFCC, MAX_FRAMES)
