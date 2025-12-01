import os
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from pathlib import Path
import json
from collections import defaultdict

from vad import get_vad_mask
from reverbiration import (
    add_reverberation,
    get_target_rt60_for_uniform_distribution,
    get_rt60_bin
)

# Конфигурация
INPUT_DIR = "/home/danya/datasets/CMU-MOSEI/Audio/WAV_16000/"
OUTPUT_DIR = "/home/danya/datasets/CMU-MOSEI/Audio/reverb/"

# Параметры VAD
VAD_PARAMS = {
    "top_db": 20,
    "frame_length": 2048,
    "hop_length": 512
}

# Диапазон RT60 (в секундах) - время реверберации
RT60_MIN = 0.1  # Минимальное значение RT60 (легкая реверберация)
RT60_MAX = 2.0  # Максимальное значение RT60 (сильная реверберация)
RT60_BINS = 30  # Количество бинов для равномерного распределения

# Диапазон wet_level (уровень реверберации, 0-1)
WET_LEVEL_MIN = 0.1  # Минимальный уровень реверберации
WET_LEVEL_MAX = 0.8  # Максимальный уровень реверберации

# Файл для отслеживания распределения RT60
DISTRIBUTION_FILE = os.path.join(OUTPUT_DIR, "rt60_distribution.json")


def load_rt60_distribution():
    """
    Загружает распределение RT60 из файла.
    
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

def save_rt60_distribution(distribution):
    """
    Сохраняет распределение RT60 в файл.
    """
    os.makedirs(os.path.dirname(DISTRIBUTION_FILE), exist_ok=True)
    with open(DISTRIBUTION_FILE, 'w') as f:
        json.dump(dict(distribution), f, indent=2)


def process_wav_file(input_path, output_path, rt60_distribution):
    """
    Обрабатывает один wav файл: добавляет реверберацию с целевым RT60.
    
    Args:
        input_path: путь к входному файлу
        output_path: путь для сохранения результата
        rt60_distribution: словарь с распределением RT60 по бинам
    
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
        
        # Выбираем целевой RT60 для равномерного распределения
        target_rt60, selected_bin = get_target_rt60_for_uniform_distribution(
            rt60_distribution, RT60_MIN, RT60_MAX, RT60_BINS
        )
        
        # Выбираем случайный wet_level в заданном диапазоне
        wet_level = np.random.uniform(WET_LEVEL_MIN, WET_LEVEL_MAX)
        
        # Добавляем реверберацию
        reverbed_audio = add_reverberation(y, sr, rt60=target_rt60, wet_level=wet_level)
        
        # Обновляем распределение
        actual_bin = get_rt60_bin(target_rt60, RT60_MIN, RT60_MAX, RT60_BINS)
        # Безопасное увеличение значения (работает и с обычным dict, и с defaultdict)
        bin_key = str(actual_bin)
        rt60_distribution[bin_key] = rt60_distribution.get(bin_key, 0) + 1
        
        # Формируем новое имя файла с RT60 и wet_level
        output_dir = os.path.dirname(output_path)
        original_filename = os.path.basename(output_path)
        # Форматируем RT60: округляем до 2 знаков после запятой, заменяем точку на _
        rt60_str = f"{target_rt60:.2f}".replace(".", "_")
        # Форматируем wet_level: округляем до 2 знаков после запятой, заменяем точку на _
        wet_str = f"{wet_level:.2f}".replace(".", "_")
        # Формируем имя файла: rt60=0_87__wet=0_5__name=original_filename.wav
        new_filename = f"rt60={rt60_str}__wet={wet_str}__name={original_filename}"
        new_output_path = os.path.join(output_dir, new_filename)
        
        # Сохраняем результат
        os.makedirs(output_dir, exist_ok=True)
        # Убеждаемся, что sr - это целое число (int)
        sr_int = int(sr)
        sf.write(new_output_path, reverbed_audio, sr_int)
        
        return True
    except Exception as e:
        import traceback
        print(f"Error processing {input_path}: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def process_dataset():
    """
    Обрабатывает все wav файлы из входной директории: добавляет реверберацию с равномерным распределением RT60.
    """
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    # Загружаем текущее распределение RT60
    rt60_distribution = load_rt60_distribution()
    
    # Находим все wav файлы (рекурсивно)
    wav_files = list(input_path.rglob("*.wav"))
    # wav_files = wav_files[:100]
    
    if len(wav_files) == 0:
        print(f"No wav files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(wav_files)} wav files to process")
    print(f"RT60 range: {RT60_MIN} - {RT60_MAX} seconds")
    print(f"Number of bins: {RT60_BINS}")
    print(f"Wet level range: {WET_LEVEL_MIN} - {WET_LEVEL_MAX}")
    
    # Обрабатываем каждый файл
    success_count = 0
    failed_count = 0
    
    for wav_file in tqdm(wav_files, desc="Processing files"):
        # Вычисляем относительный путь от входной директории
        relative_path = wav_file.relative_to(input_path)
        
        # Формируем путь для выходного файла
        output_file = output_path / relative_path
        
        # Обрабатываем файл
        if process_wav_file(str(wav_file), str(output_file), rt60_distribution):
            success_count += 1
            # Периодически сохраняем распределение
            if success_count % 100 == 0:
                save_rt60_distribution(rt60_distribution)
        else:
            failed_count += 1
    
    # Финальное сохранение распределения
    save_rt60_distribution(rt60_distribution)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"\nRT60 distribution saved to: {DISTRIBUTION_FILE}")
    
    # Выводим статистику распределения
    print("\nRT60 Distribution (files per bin):")
    bin_width = (RT60_MAX - RT60_MIN) / RT60_BINS
    for bin_idx in range(RT60_BINS):
        bin_start = RT60_MIN + bin_idx * bin_width
        bin_end = bin_start + bin_width
        count = rt60_distribution.get(str(bin_idx), 0)
        print(f"  [{bin_start:.2f}, {bin_end:.2f}) s: {count} files")

if __name__ == "__main__":
    process_dataset()

