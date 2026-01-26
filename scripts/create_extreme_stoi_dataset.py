#!/usr/bin/env python3
"""
Скрипт для создания датасета с крайними значениями STOI:
- 1000 записей с STOI ~ 1 (минимальный шум или без шума)
- 1000 записей с STOI ~ 0 (сильная реверберация)
"""

import os
import sys
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from pystoi import stoi
import random

# Добавляем путь к scripts для импорта модулей
scripts_dir = os.path.dirname(os.path.abspath(__file__))
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from vad import get_vad_mask
from noise import add_white_noise_with_target_snr
from reverbiration import add_reverberation

# Конфигурация
INPUT_DIR = "/home/danya/datasets/CMU-MOSEI/Audio/WAV_16000/"
OUTPUT_DIR = "/home/danya/datasets/CMU-MOSEI/Audio/extreme_stoi/"
SAMPLE_RATE = 16000

# Параметры для высокого STOI (0.9-1.0)
HIGH_STOI_COUNT = 1000
HIGH_STOI_SNR_MIN = 20.0  # Умеренный SNR для получения STOI в диапазоне 0.9-1.0
HIGH_STOI_SNR_MAX = 35.0
HIGH_STOI_MIN = 0.9  # Минимальный STOI для сохранения
HIGH_STOI_MAX = 1.0  # Максимальный STOI для сохранения

# Параметры для низкого STOI (0-0.1)
LOW_STOI_COUNT = 1000
LOW_STOI_RT60_MIN = 3.5  # Очень сильная реверберация
LOW_STOI_RT60_MAX = 5.0  # Еще сильнее
LOW_STOI_WET_MIN = 0.9  # Очень высокий уровень реверберации
LOW_STOI_WET_MAX = 0.98
LOW_STOI_MIN = 0.0  # Минимальный STOI для сохранения
LOW_STOI_MAX = 0.1  # Максимальный STOI для сохранения

# Параметры VAD
VAD_PARAMS = {
    "top_db": 20,
    "frame_length": 2048,
    "hop_length": 512
}


def calculate_stoi(clean, processed, sr):
    """Вычисляет STOI между чистой и обработанной записью"""
    min_len = min(len(clean), len(processed))
    clean = clean[:min_len]
    processed = processed[:min_len]
    
    # Убеждаемся, что сигналы имеют правильный тип
    clean = np.asarray(clean, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)
    
    try:
        stoi_score = stoi(clean, processed, sr, extended=False)
        return float(stoi_score)
    except Exception as e:
        print(f"Ошибка при вычислении STOI: {e}")
        return None


def create_high_stoi_files(input_files, output_dir, target_count=1000):
    """Создает файлы с высоким STOI (~1)"""
    os.makedirs(output_dir, exist_ok=True)
    
    high_stoi_files = []
    file_idx = 0
    
    print(f"\nСоздание {target_count} файлов с высоким STOI (~1)...")
    
    with tqdm(total=target_count, desc="High STOI files") as pbar:
        while len(high_stoi_files) < target_count and file_idx < len(input_files):
            input_file = input_files[file_idx % len(input_files)]
            file_idx += 1
            
            try:
                # Загружаем оригинальный файл
                y, sr = librosa.load(input_file, sr=SAMPLE_RATE, mono=True)
                
                if len(y) < SAMPLE_RATE:  # Пропускаем слишком короткие файлы
                    continue
                
                # Получаем VAD маску
                speech_mask = get_vad_mask(y, sr, **VAD_PARAMS)
                
                # Создаем несколько вариантов с разными уровнями шума
                for attempt in range(5):  # Пробуем до 5 раз для каждого файла
                    if len(high_stoi_files) >= target_count:
                        break
                    
                    # Выбираем случайный SNR в диапазоне для получения STOI 0.9-1.0
                    # С вероятностью 10% создаем файл с минимальным шумом (SNR очень высокий)
                    if random.random() < 0.1:
                        snr = np.random.uniform(45.0, 60.0)
                    else:
                        snr = np.random.uniform(HIGH_STOI_SNR_MIN, HIGH_STOI_SNR_MAX)
                    
                    processed = add_white_noise_with_target_snr(y, sr, snr, speech_mask)
                    snr_used = snr
                    
                    # Вычисляем STOI
                    stoi_score = calculate_stoi(y, processed, sr)
                    
                    if stoi_score is not None and HIGH_STOI_MIN <= stoi_score <= HIGH_STOI_MAX:  # STOI в диапазоне 0.9-1.0
                        # Сохраняем файл
                        base_name = Path(input_file).stem
                        output_filename = f"high_stoi_{stoi_score:.4f}__snr={snr_used:.2f}__name={base_name}.wav"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        sf.write(output_path, processed, sr)
                        
                        high_stoi_files.append({
                            'file': output_path,
                            'stoi': stoi_score,
                            'snr': snr_used,
                            'original': input_file
                        })
                        
                        pbar.update(1)
                        
                        if len(high_stoi_files) >= target_count:
                            break
            
            except Exception as e:
                print(f"Ошибка при обработке {input_file}: {e}")
                continue
    
    print(f"Создано {len(high_stoi_files)} файлов с высоким STOI")
    return high_stoi_files


def create_low_stoi_files(input_files, output_dir, target_count=1000):
    """Создает файлы с низким STOI (~0)"""
    os.makedirs(output_dir, exist_ok=True)
    
    low_stoi_files = []
    file_idx = 0
    
    print(f"\nСоздание {target_count} файлов с низким STOI (~0)...")
    
    with tqdm(total=target_count, desc="Low STOI files") as pbar:
        while len(low_stoi_files) < target_count and file_idx < len(input_files):
            input_file = input_files[file_idx % len(input_files)]
            file_idx += 1
            
            try:
                # Загружаем оригинальный файл
                y, sr = librosa.load(input_file, sr=SAMPLE_RATE, mono=True)
                
                if len(y) < SAMPLE_RATE:  # Пропускаем слишком короткие файлы
                    continue
                
                # Создаем несколько вариантов с разными параметрами реверберации
                for attempt in range(10):  # Пробуем до 10 раз для каждого файла
                    if len(low_stoi_files) >= target_count:
                        break
                    
                    # Выбираем случайные параметры для сильной реверберации
                    rt60 = np.random.uniform(LOW_STOI_RT60_MIN, LOW_STOI_RT60_MAX)
                    wet_level = np.random.uniform(LOW_STOI_WET_MIN, LOW_STOI_WET_MAX)
                    
                    # Добавляем реверберацию
                    processed = add_reverberation(y, sr, rt60=rt60, wet_level=wet_level)
                    
                    # Вычисляем STOI
                    stoi_score = calculate_stoi(y, processed, sr)
                    
                    if stoi_score is not None and LOW_STOI_MIN <= stoi_score <= LOW_STOI_MAX:  # STOI в диапазоне 0-0.1
                        # Сохраняем файл
                        base_name = Path(input_file).stem
                        rt60_str = f"{rt60:.2f}".replace(".", "_")
                        wet_str = f"{wet_level:.2f}".replace(".", "_")
                        output_filename = f"low_stoi_{stoi_score:.4f}__rt60={rt60_str}__wet={wet_str}__name={base_name}.wav"
                        output_path = os.path.join(output_dir, output_filename)
                        
                        sf.write(output_path, processed, sr)
                        
                        low_stoi_files.append({
                            'file': output_path,
                            'stoi': stoi_score,
                            'rt60': rt60,
                            'wet_level': wet_level,
                            'original': input_file
                        })
                        
                        pbar.update(1)
                        
                        if len(low_stoi_files) >= target_count:
                            break
            
            except Exception as e:
                print(f"Ошибка при обработке {input_file}: {e}")
                continue
    
    print(f"Создано {len(low_stoi_files)} файлов с низким STOI")
    return low_stoi_files


def plot_stoi_distribution(high_stoi_files, low_stoi_files, output_dir):
    """Строит графики распределения STOI"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Извлекаем значения STOI
    high_stoi_values = [f['stoi'] for f in high_stoi_files]
    low_stoi_values = [f['stoi'] for f in low_stoi_files]
    
    # Создаем фигуру с несколькими графиками
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Гистограмма всех значений STOI
    ax = axes[0, 0]
    ax.hist(high_stoi_values, bins=50, alpha=0.7, label=f'High STOI (n={len(high_stoi_values)})', color='green')
    ax.hist(low_stoi_values, bins=50, alpha=0.7, label=f'Low STOI (n={len(low_stoi_values)})', color='red')
    ax.set_xlabel('STOI', fontsize=12, fontweight='bold')
    ax.set_ylabel('Количество файлов', fontsize=12, fontweight='bold')
    ax.set_title('Распределение STOI для крайних значений', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Детальная гистограмма высокого STOI
    ax = axes[0, 1]
    ax.hist(high_stoi_values, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.set_xlabel('STOI', fontsize=12, fontweight='bold')
    ax.set_ylabel('Количество файлов', fontsize=12, fontweight='bold')
    ax.set_title(f'Распределение высокого STOI (mean={np.mean(high_stoi_values):.4f})', fontsize=14, fontweight='bold')
    ax.axvline(np.mean(high_stoi_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(high_stoi_values):.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Детальная гистограмма низкого STOI
    ax = axes[1, 0]
    ax.hist(low_stoi_values, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax.set_xlabel('STOI', fontsize=12, fontweight='bold')
    ax.set_ylabel('Количество файлов', fontsize=12, fontweight='bold')
    ax.set_title(f'Распределение низкого STOI (mean={np.mean(low_stoi_values):.4f})', fontsize=14, fontweight='bold')
    ax.axvline(np.mean(low_stoi_values), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(low_stoi_values):.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Box plot
    ax = axes[1, 1]
    data_to_plot = [high_stoi_values, low_stoi_values]
    bp = ax.boxplot(data_to_plot, labels=['High STOI', 'Low STOI'], patch_artist=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('red')
    ax.set_ylabel('STOI', fontsize=12, fontweight='bold')
    ax.set_title('Box plot распределения STOI', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Сохраняем график
    output_path = os.path.join(output_dir, 'extreme_stoi_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nГрафик сохранен: {output_path}")
    plt.close()
    
    # Выводим статистику
    print("\n" + "="*80)
    print("СТАТИСТИКА ДАТАСЕТА С КРАЙНИМИ ЗНАЧЕНИЯМИ STOI")
    print("="*80)
    print(f"\nВысокий STOI (~1):")
    print(f"  Количество файлов: {len(high_stoi_values)}")
    print(f"  Среднее STOI: {np.mean(high_stoi_values):.4f}")
    print(f"  Медиана STOI: {np.median(high_stoi_values):.4f}")
    print(f"  Минимум STOI: {np.min(high_stoi_values):.4f}")
    print(f"  Максимум STOI: {np.max(high_stoi_values):.4f}")
    print(f"  Стандартное отклонение: {np.std(high_stoi_values):.4f}")
    
    print(f"\nНизкий STOI (~0):")
    print(f"  Количество файлов: {len(low_stoi_values)}")
    print(f"  Среднее STOI: {np.mean(low_stoi_values):.4f}")
    print(f"  Медиана STOI: {np.median(low_stoi_values):.4f}")
    print(f"  Минимум STOI: {np.min(low_stoi_values):.4f}")
    print(f"  Максимум STOI: {np.max(low_stoi_values):.4f}")
    print(f"  Стандартное отклонение: {np.std(low_stoi_values):.4f}")
    print("="*80)


def main():
    """Основная функция"""
    print("="*80)
    print("СОЗДАНИЕ ДАТАСЕТА С КРАЙНИМИ ЗНАЧЕНИЯМИ STOI")
    print("="*80)
    
    # Проверяем существование входной директории
    if not os.path.exists(INPUT_DIR):
        print(f"Ошибка: входная директория не найдена: {INPUT_DIR}")
        return
    
    # Получаем список всех WAV файлов
    input_files = list(Path(INPUT_DIR).glob("*.wav"))
    if len(input_files) == 0:
        print(f"Ошибка: не найдено WAV файлов в {INPUT_DIR}")
        return
    
    print(f"Найдено {len(input_files)} оригинальных аудио файлов")
    
    # Перемешиваем для разнообразия
    random.shuffle(input_files)
    
    # Создаем директорию для выходных файлов
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Создаем файлы с высоким STOI
    high_stoi_files = create_high_stoi_files(input_files, OUTPUT_DIR, HIGH_STOI_COUNT)
    
    # Создаем файлы с низким STOI
    low_stoi_files = create_low_stoi_files(input_files, OUTPUT_DIR, LOW_STOI_COUNT)
    
    # Строим графики
    images_dir = os.path.join(os.path.dirname(OUTPUT_DIR), "images")
    os.makedirs(images_dir, exist_ok=True)
    plot_stoi_distribution(high_stoi_files, low_stoi_files, images_dir)
    
    print(f"\n✓ Датасет создан успешно!")
    print(f"  Высокий STOI: {len(high_stoi_files)} файлов")
    print(f"  Низкий STOI: {len(low_stoi_files)} файлов")
    print(f"  Всего: {len(high_stoi_files) + len(low_stoi_files)} файлов")
    print(f"  Сохранено в: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

