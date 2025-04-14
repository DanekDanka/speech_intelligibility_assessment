import os
import numpy as np
import soundfile as sf
import librosa

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

def add_noise_to_audio(input_path, output_path, vad_output_path=None, noise_level=0.01, vad_params=None):
    y, sr = librosa.load(input_path, sr=None, mono=True)

    if vad_params is not None:
        y_vad = energy_based_vad(y, sr, **vad_params)
        
        if vad_output_path is not None:
            os.makedirs(os.path.dirname(vad_output_path), exist_ok=True)
            sf.write(vad_output_path, y_vad, sr)
        
        y = y_vad

    noise = np.random.normal(0, noise_level, y.shape)
    noisy_audio = y + noise
    noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
    sf.write(output_path, noisy_audio, sr)

def process_directory(input_dir, output_dir, vad_dir="VAD", noise_level=0.01, vad_params=None):
    os.makedirs(output_dir, exist_ok=True)
    vad_output_dir = os.path.join(output_dir, vad_dir)
    os.makedirs(vad_output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)

        if filename.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.aiff')):
            output_path = os.path.join(output_dir, filename)
            vad_output_path = os.path.join(vad_output_dir, filename)
            
            add_noise_to_audio(
                input_path,
                output_path,
                vad_output_path=vad_output_path,
                noise_level=noise_level,
                vad_params=vad_params
            )
            
            print(f"Обработан файл: {filename}")
            print(f"  - После VAD: {vad_output_path}")
            print(f"  - С шумом: {output_path}")

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
        noise_level=0.02,
        vad_params=vad_params
    )
