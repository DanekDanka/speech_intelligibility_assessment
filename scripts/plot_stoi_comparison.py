#!/usr/bin/env python3
"""
Скрипт для построения сравнительного распределения STOI
для файлов с шумом и реверберацией (без extreme_stoi).
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import librosa
from pystoi import stoi
from scipy import stats


# Конфигурация
ORIGINAL_DIR = "/home/danya/datasets/CMU-MOSEI/Audio/WAV_16000/"
NOISE_DIR = "/home/danya/datasets/CMU-MOSEI/Audio/noise/"
REVERB_DIR = "/home/danya/datasets/CMU-MOSEI/Audio/reverb/"

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Создаем папку для сохранения графиков
IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)


def parse_noise_filename(filename):
    """
    Парсит имя файла с шумом и извлекает SNR.
    Формат: snr=12_34__name=original_filename.wav
    Или: high_stoi_...__snr=26.75__name=... или snr=inf__name=...
    """
    # Ищем snr= в любом месте строки (не только в начале)
    match = re.search(r'__snr=([0-9_.\-inf]+)__', filename)
    if not match:
        # Пробуем формат в начале строки
        match = re.search(r'^snr=([0-9_.\-inf]+)__', filename)
    
    if match:
        snr_str = match.group(1).replace('_', '.')
        # Обрабатываем случай "inf"
        if snr_str.lower() == 'inf' or snr_str.lower() == '.inf':
            return float('inf')
        try:
            return float(snr_str)
        except:
            return None
    return None


def parse_reverb_filename(filename):
    """
    Парсит имя файла с реверберацией и извлекает RT60 и wet_level.
    Формат: rt60=0_87__wet=0_5__name=original_filename.wav
    Или: low_stoi_...__rt60=4_39__wet=0_94__name=...
    """
    # Ищем rt60= в любом месте строки (не только в начале)
    # Поддерживаем как подчеркивания, так и точки в числах
    rt60_match = re.search(r'__rt60=([0-9_.]+)__', filename)
    if not rt60_match:
        # Пробуем формат в начале строки
        rt60_match = re.search(r'^rt60=([0-9_.]+)__', filename)
    # Поддерживаем как подчеркивания, так и точки в числах
    wet_match = re.search(r'__wet=([0-9_.]+)__', filename)
    
    rt60 = None
    wet = None
    
    if rt60_match:
        rt60_str = rt60_match.group(1).replace('_', '.')
        try:
            rt60 = float(rt60_str)
        except:
            pass
    
    if wet_match:
        wet_str = wet_match.group(1).replace('_', '.')
        try:
            wet = float(wet_str)
        except:
            pass
    
    return rt60, wet


def extract_original_filename(filename):
    """
    Извлекает оригинальное имя файла из конца после name=
    Формат: snr=12_34__name=original_filename.wav или rt60=0_87__wet=0_5__name=original_filename.wav
    """
    # Ищем все после последнего __name=
    match = re.search(r'__name=(.+)$', filename)
    if match:
        original_name = match.group(1)
        # Если имя уже содержит расширение, возвращаем как есть, иначе добавляем .wav
        if not original_name.endswith('.wav'):
            return original_name + '.wav'
        return original_name
    # Fallback: если формат старый, извлекаем после последнего __
    if '__' in filename:
        return filename.split('__')[-1]
    return filename


def find_original_file(original_filename, base_dir):
    """
    Находит оригинальный файл в директории (рекурсивно).
    """
    base_path = Path(base_dir)
    for wav_file in base_path.rglob(original_filename):
        return str(wav_file)
    return None


def calculate_stoi_score(processed_file, original_file):
    """
    Вычисляет STOI между обработанным и оригинальным файлом.
    """
    try:
        # Загружаем оба файла
        processed, sr_proc = librosa.load(processed_file, sr=None, mono=True)
        original, sr_orig = librosa.load(original_file, sr=None, mono=True)
        
        # Убеждаемся, что частота дискретизации одинакова
        if sr_proc != sr_orig:
            # Ресемплируем к меньшей частоте
            target_sr = min(sr_proc, sr_orig)
            if sr_proc != target_sr:
                processed = librosa.resample(processed, orig_sr=sr_proc, target_sr=target_sr)
            if sr_orig != target_sr:
                original = librosa.resample(original, orig_sr=sr_orig, target_sr=target_sr)
            sr = target_sr
        else:
            sr = sr_proc
        
        # Обрезаем до минимальной длины
        min_len = min(len(processed), len(original))
        processed = processed[:min_len]
        original = original[:min_len]
        
        # Вычисляем STOI
        stoi_score = stoi(original, processed, sr, extended=False)
        return stoi_score
    except Exception as e:
        print(f"Error calculating STOI for {processed_file}: {e}")
        return None


def main():
    print("Собираем данные о файлах с шумом...")
    noise_data = []
    
    # Обрабатываем файлы из папки noise (НЕ из extreme_stoi)
    noise_path = Path(NOISE_DIR)
    if noise_path.exists():
        noise_files = list(noise_path.rglob("*.wav"))
        print(f"Найдено {len(noise_files)} файлов с шумом в папке noise")
        
        for noise_file in tqdm(noise_files, desc="Обработка файлов с шумом"):
            filename = noise_file.name
            snr = parse_noise_filename(filename)
            
            if snr is not None:
                original_filename = extract_original_filename(filename)
                original_file = find_original_file(original_filename, ORIGINAL_DIR)
                
                if original_file:
                    stoi_score = calculate_stoi_score(str(noise_file), original_file)
                    if stoi_score is not None:
                        noise_data.append({
                            'filename': str(noise_file),
                            'snr': snr,
                            'stoi': stoi_score,
                            'original_file': original_file
                        })
                else:
                    print(f"Не найден оригинальный файл для {filename}")
            else:
                print(f"Не удалось распарсить SNR из {filename}")
    else:
        print(f"Директория {NOISE_DIR} не существует")
    
    print(f"\nОбработано {len(noise_data)} файлов с шумом")
    
    print("\nСобираем данные о файлах с реверберацией...")
    reverb_data = []
    
    # Обрабатываем файлы из папки reverb (НЕ из extreme_stoi)
    reverb_path = Path(REVERB_DIR)
    if reverb_path.exists():
        reverb_files = list(reverb_path.rglob("*.wav"))
        print(f"Найдено {len(reverb_files)} файлов с реверберацией в папке reverb")
        
        for reverb_file in tqdm(reverb_files, desc="Обработка файлов с реверберацией"):
            filename = reverb_file.name
            rt60, wet = parse_reverb_filename(filename)
            
            if rt60 is not None and wet is not None:
                original_filename = extract_original_filename(filename)
                original_file = find_original_file(original_filename, ORIGINAL_DIR)
                
                if original_file:
                    stoi_score = calculate_stoi_score(str(reverb_file), original_file)
                    if stoi_score is not None:
                        reverb_data.append({
                            'filename': str(reverb_file),
                            'rt60': rt60,
                            'wet_level': wet,
                            'stoi': stoi_score,
                            'original_file': original_file
                        })
                else:
                    print(f"Не найден оригинальный файл для {filename}")
            else:
                print(f"Не удалось распарсить параметры из {filename}")
    else:
        print(f"Директория {REVERB_DIR} не существует")
    
    print(f"\nОбработано {len(reverb_data)} файлов с реверберацией")
    
    # Создаем DataFrame
    df_noise = pd.DataFrame(noise_data)
    df_reverb = pd.DataFrame(reverb_data)
    
    # Фильтруем значения STOI от 0 до 1
    if len(df_noise) > 0:
        df_noise = df_noise[(df_noise['stoi'] >= 0) & (df_noise['stoi'] <= 1)]
        print(f"\nПосле фильтрации STOI [0, 1]: {len(df_noise)} файлов с шумом")
    
    if len(df_reverb) > 0:
        df_reverb = df_reverb[(df_reverb['stoi'] >= 0) & (df_reverb['stoi'] <= 1)]
        print(f"После фильтрации STOI [0, 1]: {len(df_reverb)} файлов с реверберацией")
    
    # Строим сравнительный график STOI
    if len(df_noise) > 0 and len(df_reverb) > 0:
        # 1) Гистограммы (отдельная картинка)
        fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        ax1.hist(df_noise['stoi'], bins=30, edgecolor='black', alpha=0.6, color='green',
                 label='С шумом', density=True)
        ax1.hist(df_reverb['stoi'], bins=30, edgecolor='black', alpha=0.6, color='purple',
                 label='С реверберацией', density=True)
        ax1.set_xlabel('STOI', fontsize=12)
        ax1.set_ylabel('Плотность вероятности', fontsize=12)
        ax1.set_title('Сравнительное распределение STOI', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(bottom=0)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        plt.tight_layout()
        save_path1 = os.path.join(IMAGES_DIR, "stoi_comparison.png")
        plt.savefig(save_path1, dpi=300, bbox_inches='tight')
        print(f"\nГрафик (гистограмма) сохранён: {save_path1}")
        plt.close(fig1)

        # 2) Линейный график KDE — отдельная картинка, удобная для печати в ЧБ
        # Разные типы линий и маркеры, чтобы в ч/б было понятно, что есть что
        x_line = np.linspace(0, 1, 200)
        kde_noise = stats.gaussian_kde(df_noise['stoi'])
        kde_reverb = stats.gaussian_kde(df_reverb['stoi'])

        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        ax2.plot(x_line, kde_noise(x_line), color='black', linestyle='-', linewidth=2,
                 label='С шумом', marker='o', markevery=20, markersize=4)
        ax2.plot(x_line, kde_reverb(x_line), color='0.4', linestyle='--', linewidth=2,
                 label='С реверберацией', marker='s', markevery=20, markersize=4)
        ax2.set_xlabel('STOI', fontsize=12)
        ax2.set_ylabel('Плотность вероятности', fontsize=12)
        ax2.set_title('Сравнительное распределение STOI (KDE)', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 1)
        ax2.set_ylim(bottom=0)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        plt.tight_layout()
        save_path2 = os.path.join(IMAGES_DIR, "stoi_comparison_line.png")
        plt.savefig(save_path2, dpi=300, bbox_inches='tight')
        print(f"График (линия, для ЧБ) сохранён: {save_path2}")
        plt.close(fig2)
        plt.show()
    else:
        if len(df_noise) == 0:
            print("Нет данных о файлах с шумом для построения графиков")
        if len(df_reverb) == 0:
            print("Нет данных о файлах с реверберацией для построения графиков")


if __name__ == "__main__":
    main()
