import os
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from pathlib import Path
import json
from collections import defaultdict

from vad import get_vad_mask
from noise import (
    add_white_noise_with_target_snr,
    get_target_snr_for_uniform_distribution,
    get_snr_bin
)
from reverbiration import (
    add_reverberation,
    get_target_rt60_for_uniform_distribution,
    get_rt60_bin
)

# Конфигурация
INPUT_DIR = "/home/danya/datasets/CMU-MOSEI/Audio/WAV_16000/"
OUTPUT_DIR = "/home/danya/datasets/CMU-MOSEI/Audio/noise_reverb/"

# Параметры VAD
VAD_PARAMS = {
    "top_db": 20,
    "frame_length": 2048,
    "hop_length": 512
}

# Диапазон SNR (в dB)
SNR_MIN = -10.0  # Минимальное значение SNR
SNR_MAX = 20.0  # Максимальное значение SNR
SNR_BINS = 30  # Количество бинов для равномерного распределения

# Диапазон RT60 (в секундах) - время реверберации
RT60_MIN = 0.1  # Минимальное значение RT60 (легкая реверберация)
RT60_MAX = 2.0  # Максимальное значение RT60 (сильная реверберация)
RT60_BINS = 30  # Количество бинов для равномерного распределения

# Диапазон wet_level (уровень реверберации, 0-1)
WET_LEVEL_MIN = 0.1  # Минимальный уровень реверберации
WET_LEVEL_MAX = 0.8  # Максимальный уровень реверберации

# Файл для отслеживания распределения
DISTRIBUTION_FILE = os.path.join(OUTPUT_DIR, "distribution.json")


def load_distribution():
    """Загружает распределение из файла."""
    if os.path.exists(DISTRIBUTION_FILE):
        with open(DISTRIBUTION_FILE, 'r') as f:
            loaded_dict = json.load(f)
            result = defaultdict(int)
            result.update(loaded_dict)
            return result
    return defaultdict(int)

def save_distribution(distribution):
    """Сохраняет распределение в файл."""
    os.makedirs(os.path.dirname(DISTRIBUTION_FILE), exist_ok=True)
    with open(DISTRIBUTION_FILE, 'w') as f:
        json.dump(dict(distribution), f, indent=2)


def process_wav_file(input_path, output_path, distribution):
    """
    Обрабатывает один wav файл: добавляет шум и реверберацию.
    """
    try:
        # Загружаем аудио
        audio_data = librosa.load(input_path, sr=None, mono=True)
        
        if not isinstance(audio_data, tuple) or len(audio_data) != 2:
            print(f"Warning: Unexpected audio format in {input_path}, skipping...")
            return False
        
        y, sr = audio_data
        
        sr = float(sr)
        if sr <= 0:
            print(f"Warning: Invalid sample rate in {input_path}, skipping...")
            return False
        
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.flatten()
        
        if len(y) == 0:
            print(f"Warning: Empty audio file {input_path}, skipping...")
            return False
        
        # Получаем маску VAD
        speech_mask = get_vad_mask(y, sr, **VAD_PARAMS)
        speech_mask = np.asarray(speech_mask, dtype=bool)
        
        if not np.any(speech_mask):
            print(f"Warning: No speech detected in {input_path}, skipping...")
            return False
        
        # Преобразуем распределение для использования с функциями из модулей
        # Функции ожидают ключи в формате "0", "1", но мы используем "snr_0", "rt60_0"
        snr_distribution = {}
        rt60_distribution = {}
        for key, value in distribution.items():
            if key.startswith("snr_"):
                bin_idx = key.replace("snr_", "")
                snr_distribution[bin_idx] = value
            elif key.startswith("rt60_"):
                bin_idx = key.replace("rt60_", "")
                rt60_distribution[bin_idx] = value
        
        # Выбираем целевые параметры для равномерного распределения
        target_snr, snr_bin = get_target_snr_for_uniform_distribution(
            snr_distribution, SNR_MIN, SNR_MAX, SNR_BINS
        )
        target_rt60, rt60_bin = get_target_rt60_for_uniform_distribution(
            rt60_distribution, RT60_MIN, RT60_MAX, RT60_BINS
        )
        
        # Выбираем случайный wet_level
        wet_level = np.random.uniform(WET_LEVEL_MIN, WET_LEVEL_MAX)
        
        # Сначала добавляем шум
        noisy_audio = add_white_noise_with_target_snr(y, sr, target_snr, speech_mask)
        
        # Затем добавляем реверберацию к зашумленному сигналу
        processed_audio = add_reverberation(noisy_audio, sr, rt60=target_rt60, wet_level=wet_level)
        
        # Обновляем распределение
        actual_snr_bin = get_snr_bin(target_snr, SNR_MIN, SNR_MAX, SNR_BINS)
        actual_rt60_bin = get_rt60_bin(target_rt60, RT60_MIN, RT60_MAX, RT60_BINS)
        
        distribution[f"snr_{actual_snr_bin}"] = distribution.get(f"snr_{actual_snr_bin}", 0) + 1
        distribution[f"rt60_{actual_rt60_bin}"] = distribution.get(f"rt60_{actual_rt60_bin}", 0) + 1
        
        # Формируем новое имя файла
        output_dir = os.path.dirname(output_path)
        original_filename = os.path.basename(output_path)
        
        # Форматируем параметры
        snr_str = f"{target_snr:.2f}".replace(".", "_")
        rt60_str = f"{target_rt60:.2f}".replace(".", "_")
        wet_str = f"{wet_level:.2f}".replace(".", "_")
        
        # Формируем имя файла: snr=12_34__rt60=0_87__wet=0_5__name=original_filename.wav
        new_filename = f"snr={snr_str}__rt60={rt60_str}__wet={wet_str}__name={original_filename}"
        new_output_path = os.path.join(output_dir, new_filename)
        
        # Сохраняем результат
        os.makedirs(output_dir, exist_ok=True)
        sr_int = int(sr)
        sf.write(new_output_path, processed_audio, sr_int)
        
        return True
    except Exception as e:
        import traceback
        print(f"Error processing {input_path}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def process_dataset():
    """
    Обрабатывает все wav файлы из входной директории: добавляет шум и реверберацию.
    """
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    # Загружаем текущее распределение
    distribution = load_distribution()
    
    # Находим все wav файлы (рекурсивно)
    wav_files = list(input_path.rglob("*.wav"))
    # wav_files = wav_files[:100]
    
    if len(wav_files) == 0:
        print(f"No wav files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(wav_files)} wav files to process")
    print(f"SNR range: {SNR_MIN} - {SNR_MAX} dB")
    print(f"RT60 range: {RT60_MIN} - {RT60_MAX} seconds")
    print(f"Wet level range: {WET_LEVEL_MIN} - {WET_LEVEL_MAX}")
    
    # Обрабатываем каждый файл
    success_count = 0
    failed_count = 0
    
    for wav_file in tqdm(wav_files, desc="Processing files"):
        relative_path = wav_file.relative_to(input_path)
        output_file = output_path / relative_path
        
        if process_wav_file(str(wav_file), str(output_file), distribution):
            success_count += 1
            if success_count % 100 == 0:
                save_distribution(distribution)
        else:
            failed_count += 1
    
    # Финальное сохранение распределения
    save_distribution(distribution)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"\nDistribution saved to: {DISTRIBUTION_FILE}")

if __name__ == "__main__":
    process_dataset()

