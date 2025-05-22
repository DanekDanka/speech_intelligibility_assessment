import os
import numpy as np
import soundfile as sf
import librosa
import random
from tqdm import tqdm
from scipy.signal import convolve
from pystoi import stoi
import pandas as pd
import shutil

# Конфигурация
class Config:
    INPUT_DIR = "/home/danya/develop/datasets/CMU-MOSEI/Audio/WAV_16000/"
    OUTPUT_DIR = "/home/danya/develop/datasets/CMU-MOSEI/Audio/balanced_stoi_dataset/"
    VAD_PARAMS = {
        "top_db": 20,
        "frame_length": 2048,
        "hop_length": 512
    }
    SNR_RANGES = [
        (20, 30),   # Высокое SNR (высокий STOI)
        (10, 20),   # Среднее SNR
        (0, 10)     # Низкое SNR (низкий STOI)
    ]
    REVERB_PARAMS = [
        {"rt60": 0.3, "wet_level": 0.2},  # Легкая реверберация
        {"rt60": 0.8, "wet_level": 0.5},  # Средняя реверберация
        {"rt60": 1.5, "wet_level": 0.8}   # Сильная реверберация
    ]
    TARGET_SAMPLES_PER_BIN = 1000  # Количество образцов в каждом STOI бине
    STOI_BINS = 10  # Количество бинов для STOI (0-1 с шагом 0.1)

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
    rms = np.sqrt(np.mean(y**2))
    return librosa.amplitude_to_db([rms], ref=1.0)[0]

def add_noise_with_variable_snr(y, sr, target_snr):
    signal_db = calculate_rms_db(y)
    noise_db = signal_db - target_snr
    noise_level = librosa.db_to_amplitude(noise_db, ref=1.0)
    noise = np.random.normal(0, noise_level, len(y))
    noisy_audio = y + noise
    noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
    return noisy_audio

def add_reverberation(y, sr, rt60=0.8, wet_level=0.3):
    length = int(rt60 * sr)
    ir = np.random.normal(0, 1, length)
    ir = np.exp(-np.linspace(0, 10, length)) * ir
    ir = ir / np.max(np.abs(ir))
    
    wet = convolve(y, ir, mode='same')
    dry_level = 1 - wet_level
    processed = dry_level * y + wet_level * wet
    processed = processed / np.max(np.abs(processed))
    return processed

def calculate_stoi(clean, noisy, sr):
    min_len = min(len(clean), len(noisy))
    clean = clean[:min_len]
    noisy = noisy[:min_len]
    return stoi(clean, noisy, sr, extended=False)

def process_audio_file(input_path, output_dir, file_counter):
    y, sr = librosa.load(input_path, sr=None, mono=True)
    
    # Применяем VAD
    y_vad = energy_based_vad(y, sr, **Config.VAD_PARAMS)
    
    # Создаем различные версии аудио
    versions = []
    
    # 1. Оригинал после VAD (высокий STOI)
    versions.append(("vad_only", y_vad))
    
    # 2. Только шум
    for snr_range in Config.SNR_RANGES:
        target_snr = random.uniform(snr_range[0], snr_range[1])
        noisy = add_noise_with_variable_snr(y_vad, sr, target_snr)
        versions.append((f"noise_snr_{target_snr:.1f}", noisy))
    
    # 3. Только реверберация
    for reverb in Config.REVERB_PARAMS:
        reverbed = add_reverberation(y_vad, sr, **reverb)
        versions.append((f"reverb_rt60_{reverb['rt60']}_wet_{reverb['wet_level']}", reverbed))
    
    # 4. Комбинации шума и реверберации
    for snr_range in Config.SNR_RANGES:
        for reverb in Config.REVERB_PARAMS:
            target_snr = random.uniform(snr_range[0], snr_range[1])
            noisy = add_noise_with_variable_snr(y_vad, sr, target_snr)
            noisy_reverbed = add_reverberation(noisy, sr, **reverb)
            versions.append((f"noise_snr_{target_snr:.1f}_reverb_rt60_{reverb['rt60']}_wet_{reverb['wet_level']}", 
                           noisy_reverbed))
    
    # Сохраняем версии и вычисляем STOI
    results = []
    base_filename = os.path.splitext(os.path.basename(input_path))[0]
    
    for i, (version_name, processed_audio) in enumerate(versions):
        output_filename = f"{base_filename}_{version_name}_{file_counter}.wav"
        output_path = os.path.join(output_dir, "audio", output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        sf.write(output_path, processed_audio, sr)
        
        # Вычисляем STOI относительно оригинала после VAD
        stoi_score = calculate_stoi(y_vad, processed_audio, sr)
        
        results.append({
            "filename": output_filename,
            "version": version_name,
            "stoi": stoi_score,
            "original": base_filename + ".wav"
        })
    
    return results

def create_balanced_dataset():
    # Создаем выходные директории
    os.makedirs(os.path.join(Config.OUTPUT_DIR, "audio"), exist_ok=True)
    csv_path = os.path.join(Config.OUTPUT_DIR, "metadata.csv")
    
    # Получаем список аудиофайлов
    audio_files = [f for f in os.listdir(Config.INPUT_DIR) 
                 if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.aiff'))]
    
    # Обрабатываем файлы и собираем метаданные
    all_results = []
    file_counter = 0
    
    for filename in tqdm(audio_files, desc="Processing original files"):
        input_path = os.path.join(Config.INPUT_DIR, filename)
        results = process_audio_file(input_path, Config.OUTPUT_DIR, file_counter)
        all_results.extend(results)
        file_counter += 1
    
    # Создаем DataFrame с метаданными
    df = pd.DataFrame(all_results)
    
    # Балансируем датасет по STOI бинам
    balanced_files = []
    bin_edges = np.linspace(0, 1, Config.STOI_BINS + 1)
    
    for i in range(Config.STOI_BINS):
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        
        # Выбираем файлы из текущего бина
        bin_files = df[(df['stoi'] >= lower) & (df['stoi'] < upper)]
        
        # Если файлов недостаточно, используем повторения
        if len(bin_files) < Config.TARGET_SAMPLES_PER_BIN:
            repeats = (Config.TARGET_SAMPLES_PER_BIN // len(bin_files)) + 1
            bin_files = pd.concat([bin_files]*repeats, ignore_index=True)
        
        # Выбираем нужное количество случайных файлов
        balanced_files.append(bin_files.sample(Config.TARGET_SAMPLES_PER_BIN, replace=True))
    
    # Объединяем сбалансированные данные
    balanced_df = pd.concat(balanced_files, ignore_index=True)
    
    # Сохраняем метаданные
    balanced_df.to_csv(csv_path, index=False)
    
    print(f"Created balanced dataset with {len(balanced_df)} files.")
    print(f"STOI distribution:")
    print(balanced_df['stoi'].describe())
    
    # Визуализация распределения STOI
    plt.figure(figsize=(10, 6))
    plt.hist(balanced_df['stoi'], bins=Config.STOI_BINS)
    plt.title("STOI Distribution in Balanced Dataset")
    plt.xlabel("STOI score")
    plt.ylabel("Number of samples")
    plt.savefig(os.path.join(Config.OUTPUT_DIR, "stoi_distribution.png"))
    plt.close()
    
    return balanced_df

if __name__ == "__main__":
    df = create_balanced_dataset()
