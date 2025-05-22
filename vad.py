import os
import numpy as np
import soundfile as sf
import librosa
import random
from tqdm import tqdm

def energy_based_vad(y, sr, top_db=20, frame_length=2048, hop_length=512):
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    energy = np.sum(frames**2, axis=0)
    energy_db = librosa.amplitude_to_db(energy, ref=np.max)
    speech_frames = energy_db > -top_db

    mask = np.zeros_like(y, dtype=bool)
    for i, is_speech in enumerate(speech_frames):
        if is_speech:
            start = i * hop_length
            end = start + frame_length
            mask[start:end] = True

    return y[mask]

def calculate_rms_db(y):
    """Вычисляет RMS уровень сигнала в dB"""
    rms = np.sqrt(np.mean(y**2))
    return librosa.amplitude_to_db([rms], ref=1.0)[0]

def add_noise_with_variable_snr(y, sr, target_snr_range=(5, 30)):
    """Добавляет шум с переменным ОСШ в заданном диапазоне"""
    signal_db = calculate_rms_db(y)
    target_snr = random.uniform(target_snr_range[0], target_snr_range[1])
    noise_db = signal_db - target_snr
    noise_level = librosa.db_to_amplitude(noise_db, ref=1.0)
    noise = np.random.normal(0, noise_level, len(y))
    noisy_audio = y + noise
    noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
    return noisy_audio, target_snr

def add_noise_to_audio(input_path, output_path, vad_output_path=None, 
                     snr_range=(5, 30), vad_params=None):
    y, sr = librosa.load(input_path, sr=None, mono=True)

    if vad_params is not None:
        y_vad = energy_based_vad(y, sr, **vad_params)
        if vad_output_path is not None:
            os.makedirs(os.path.dirname(vad_output_path), exist_ok=True)
            sf.write(vad_output_path, y_vad, sr)
        y = y_vad

    noisy_audio, snr_used = add_noise_with_variable_snr(y, sr, target_snr_range=snr_range)
    sf.write(output_path, noisy_audio, sr)
    return snr_used

def process_directory(input_dir, output_dir, vad_dir="VAD", 
                     snr_range=(5, 30), vad_params=None):
    os.makedirs(output_dir, exist_ok=True)
    vad_output_dir = os.path.join(output_dir, vad_dir)
    os.makedirs(vad_output_dir, exist_ok=True)

    audio_files = [f for f in os.listdir(input_dir) 
                 if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.aiff'))]

    for filename in tqdm(audio_files, desc="Processing audio files"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        vad_output_path = os.path.join(vad_output_dir, filename)
        
        add_noise_to_audio(
            input_path,
            output_path,
            vad_output_path=vad_output_path,
            snr_range=snr_range,
            vad_params=vad_params
        )

if __name__ == "__main__":
    input_directory = "/home/danya/develop/datasets/CMU-MOSEI/Audio/WAV_16000/"
    output_directory = "/home/danya/develop/datasets/CMU-MOSEI/Audio/vad_noise/"
    vad_directory = "/home/danya/develop/datasets/CMU-MOSEI/Audio/vad/"

    vad_params = {
        "top_db": 20,
        "frame_length": 2048,
        "hop_length": 512
    }

    process_directory(
        input_directory,
        output_directory,
        vad_dir=vad_directory,
        snr_range=(5, 20),
        vad_params=vad_params
    )
