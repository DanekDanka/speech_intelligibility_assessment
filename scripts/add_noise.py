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
    calculate_power,
    calculate_snr_from_powers,
    add_white_noise_with_target_snr,
    get_target_snr_for_uniform_distribution,
    get_snr_bin
)

# Конфигурация
INPUT_DIR = "/home/danya/datasets/CMU-MOSEI/Audio/WAV_16000/"
OUTPUT_DIR = "/home/danya/datasets/CMU-MOSEI/Audio/noise/"

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

# Файл для отслеживания распределения SNR
DISTRIBUTION_FILE = os.path.join(OUTPUT_DIR, "snr_distribution.json")


def load_snr_distribution():
    """
    Загружает распределение SNR из файла.
    
    Returns:
        Словарь с распределением по бинам (defaultdict для безопасного доступа)
    """
    if os.path.exists(DISTRIBUTION_FILE):
        with open(DISTRIBUTION_FILE, 'r') as f:
            loaded_dict = json.load(f)
            # Преобразуем в defaultdict для безопасного доступа
            result = defaultdict(int)
            result.update(loaded_dict)
            return result
    return defaultdict(int)

def save_snr_distribution(distribution):
    """
    Сохраняет распределение SNR в файл.
    """
    os.makedirs(os.path.dirname(DISTRIBUTION_FILE), exist_ok=True)
    with open(DISTRIBUTION_FILE, 'w') as f:
        json.dump(dict(distribution), f, indent=2)


def process_wav_file(input_path, output_path, snr_distribution):
    """
    Обрабатывает один wav файл: добавляет белый шум с целевым SNR.
    
    Args:
        input_path: путь к входному файлу
        output_path: путь для сохранения результата
        snr_distribution: словарь с распределением SNR по бинам
    
    Returns:
        True если успешно, False иначе
    """
    try:
        # Загружаем аудио
        audio_data = librosa.load(input_path, sr=None, mono=True)
        
        # Проверяем, что librosa.load вернул кортеж
        if not isinstance(audio_data, tuple) or len(audio_data) != 2:
            print(f"Warning: Unexpected audio format in {input_path}, skipping...")
            return False
        
        y, sr = audio_data
        
        # Убеждаемся, что sr - это число
        sr = float(sr)
        if sr <= 0:
            print(f"Warning: Invalid sample rate in {input_path}, skipping...")
            return False
        
        # Убеждаемся, что y - это numpy array (не список) и 1D
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.flatten()
        
        # Проверяем, что сигнал не пустой
        if len(y) == 0:
            print(f"Warning: Empty audio file {input_path}, skipping...")
            return False
        
        # Получаем маску VAD
        speech_mask = get_vad_mask(y, sr, **VAD_PARAMS)
        
        # Убеждаемся, что speech_mask - это numpy array
        speech_mask = np.asarray(speech_mask, dtype=bool)
        
        # Проверяем, есть ли речь в файле
        if not np.any(speech_mask):
            print(f"Warning: No speech detected in {input_path}, skipping...")
            return False
        
        # Выбираем целевой SNR для равномерного распределения
        target_snr, selected_bin = get_target_snr_for_uniform_distribution(
            snr_distribution, SNR_MIN, SNR_MAX, SNR_BINS
        )
        
        # Вычисляем исходную мощность речи (для проверки SNR)
        original_speech_power = calculate_power(y[speech_mask])
        
        # Добавляем белый шум
        noisy_audio = add_white_noise_with_target_snr(y, sr, target_snr, speech_mask)
        
        # Проверяем фактический SNR
        # Вычисляем мощность добавленного шума (разница между зашумленным и исходным сигналом)
        noise_signal = noisy_audio - y
        noise_power = calculate_power(noise_signal)
        actual_snr = calculate_snr_from_powers(original_speech_power, noise_power)
        
        # Обновляем распределение
        actual_bin = get_snr_bin(actual_snr, SNR_MIN, SNR_MAX, SNR_BINS)
        # Безопасное увеличение значения (работает и с обычным dict, и с defaultdict)
        bin_key = str(actual_bin)
        snr_distribution[bin_key] = snr_distribution.get(bin_key, 0) + 1
        
        # Формируем новое имя файла с SNR
        output_dir = os.path.dirname(output_path)
        original_filename = os.path.basename(output_path)
        # Форматируем SNR: округляем до 2 знаков после запятой, заменяем точку на _
        snr_str = f"{actual_snr:.2f}".replace(".", "_")
        # Формируем имя файла: snr=12_34__name=original_filename.wav
        new_filename = f"snr={snr_str}__name={original_filename}"
        new_output_path = os.path.join(output_dir, new_filename)
        
        # Сохраняем результат
        os.makedirs(output_dir, exist_ok=True)
        # Убеждаемся, что sr - это целое число (int)
        sr_int = int(sr)
        sf.write(new_output_path, noisy_audio, sr_int)
        
        return True
    except Exception as e:
        import traceback
        print(f"Error processing {input_path}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def process_dataset():
    """
    Обрабатывает все wav файлы из входной директории: добавляет белый шум с равномерным распределением SNR.
    """
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    # Загружаем текущее распределение SNR
    snr_distribution = load_snr_distribution()
    
    # Находим все wav файлы (рекурсивно)
    wav_files = list(input_path.rglob("*.wav"))
    # wav_files = wav_files[:100]
    
    if len(wav_files) == 0:
        print(f"No wav files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(wav_files)} wav files to process")
    print(f"SNR range: {SNR_MIN} - {SNR_MAX} dB")
    print(f"Number of bins: {SNR_BINS}")
    
    # Обрабатываем каждый файл
    success_count = 0
    failed_count = 0
    
    for wav_file in tqdm(wav_files, desc="Processing files"):
        # Вычисляем относительный путь от входной директории
        relative_path = wav_file.relative_to(input_path)
        
        # Формируем путь для выходного файла
        output_file = output_path / relative_path
        
        # Обрабатываем файл
        if process_wav_file(str(wav_file), str(output_file), snr_distribution):
            success_count += 1
            # Периодически сохраняем распределение
            if success_count % 100 == 0:
                save_snr_distribution(snr_distribution)
        else:
            failed_count += 1
    
    # Финальное сохранение распределения
    save_snr_distribution(snr_distribution)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"\nSNR distribution saved to: {DISTRIBUTION_FILE}")
    
    # Выводим статистику распределения
    print("\nSNR Distribution (files per bin):")
    bin_width = (SNR_MAX - SNR_MIN) / SNR_BINS
    for bin_idx in range(SNR_BINS):
        bin_start = SNR_MIN + bin_idx * bin_width
        bin_end = bin_start + bin_width
        count = snr_distribution.get(str(bin_idx), 0)
        print(f"  [{bin_start:.1f}, {bin_end:.1f}) dB: {count} files")

if __name__ == "__main__":
    process_dataset()

