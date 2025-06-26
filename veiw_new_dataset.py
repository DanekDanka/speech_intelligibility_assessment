import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import soundfile as sf
from pystoi import stoi

# Конфигурация
class Config:
    OUTPUT_DIR = "/home/danya/develop/datasets/CMU-MOSEI/Audio/balanced_stoi_dataset/"
    IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")
    METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.csv")

def load_and_analyze_dataset():
    # Создаем папку для изображений
    os.makedirs(Config.IMAGES_DIR, exist_ok=True)
    
    # Загружаем метаданные
    df = pd.read_csv(Config.METADATA_PATH)
    
    # 1. Распределение STOI
    plt.figure(figsize=(12, 6))
    sns.histplot(df['stoi'], bins=20, kde=True)
    plt.title('Distribution of STOI Scores in Dataset')
    plt.xlabel('STOI Score')
    plt.ylabel('Count')
    plt.savefig(os.path.join(Config.IMAGES_DIR, 'stoi_distribution.png'))
    plt.close()
    
    # 2. Распределение по типам обработки
    plt.figure(figsize=(12, 6))
    df['version_type'] = df['version'].apply(lambda x: x.split('_')[0])
    sns.countplot(data=df, y='version_type', order=df['version_type'].value_counts().index)
    plt.title('Count of Different Processing Types')
    plt.xlabel('Count')
    plt.ylabel('Processing Type')
    plt.savefig(os.path.join(Config.IMAGES_DIR, 'processing_types_distribution.png'))
    plt.close()
    
    # 3. Коробчатая диаграмма STOI по типам обработки
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='stoi', y='version_type')
    plt.title('STOI Distribution by Processing Type')
    plt.xlabel('STOI Score')
    plt.ylabel('Processing Type')
    plt.savefig(os.path.join(Config.IMAGES_DIR, 'stoi_by_processing_type.png'))
    plt.close()
    
    # 4. Проверка корреляции между параметрами и STOI
    # Извлекаем параметры из названий версий
    def extract_params(row):
        params = {}
        parts = row['version'].split('_')
        
        if 'snr' in parts:
            snr_idx = parts.index('snr') + 1
            params['snr'] = float(parts[snr_idx])
        
        if 'rt60' in parts:
            rt60_idx = parts.index('rt60') + 1
            params['rt60'] = float(parts[rt60_idx])
        
        if 'wet' in parts:
            wet_idx = parts.index('wet') + 1
            params['wet_level'] = float(parts[wet_idx])
        
        return pd.Series(params)
    
    params_df = df.join(df.apply(extract_params, axis=1))
    
    # Графики зависимости STOI от параметров
    if 'snr' in params_df.columns:
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=params_df, x='snr', y='stoi', alpha=0.5)
        plt.title('STOI vs SNR')
        plt.savefig(os.path.join(Config.IMAGES_DIR, 'stoi_vs_snr.png'))
        plt.close()
    
    if 'rt60' in params_df.columns:
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=params_df, x='rt60', y='stoi', alpha=0.5)
        plt.title('STOI vs RT60')
        plt.savefig(os.path.join(Config.IMAGES_DIR, 'stoi_vs_rt60.png'))
        plt.close()
    
    if 'wet_level' in params_df.columns:
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=params_df, x='wet_level', y='stoi', alpha=0.5)
        plt.title('STOI vs Wet Level')
        plt.savefig(os.path.join(Config.IMAGES_DIR, 'stoi_vs_wet_level.png'))
        plt.close()
    
    # 5. Проверка случайных файлов (аудио и STOI)
    print("\nVerifying random samples from dataset...")
    sample_files = df.sample(5)['filename'].values
    
    for filename in tqdm(sample_files, desc="Verifying audio files"):
        filepath = os.path.join(Config.OUTPUT_DIR, "audio", filename)
        
        try:
            # Проверяем, что файл существует и может быть прочитан
            audio, sr = sf.read(filepath)
            
            # Проверяем, что аудио не пустое
            assert len(audio) > 0, f"Empty audio file: {filename}"
            
            # Проверяем, что значения в допустимом диапазоне
            assert np.max(np.abs(audio)) <= 1.0, f"Audio clipping in file: {filename}"
            
            # Проверяем соответствие STOI в метаданных
            original_filename = df[df['filename'] == filename]['original'].values[0]
            original_path = os.path.join(Config.OUTPUT_DIR.replace("balanced_stoi_dataset", ""), original_filename)
            
            if os.path.exists(original_path):
                original_audio, original_sr = sf.read(original_path)
                original_audio = original_audio[:len(audio)]  # Обрезаем до одинаковой длины
                
                # Вычисляем STOI
                calculated_stoi = stoi(original_audio, audio, original_sr, extended=False)
                metadata_stoi = df[df['filename'] == filename]['stoi'].values[0]
                
                # Проверяем, что значения STOI близки (с учетом возможного округления)
                assert abs(calculated_stoi - metadata_stoi) < 0.05, \
                    f"STOI mismatch for {filename}: metadata={metadata_stoi}, calculated={calculated_stoi}"
            
        except Exception as e:
            print(f"\nError in file {filename}: {str(e)}")
    
    print("\nDataset verification completed!")
    
    # Сохраняем сводную статистику
    stats = {
        "total_files": len(df),
        "stoi_mean": df['stoi'].mean(),
        "stoi_std": df['stoi'].std(),
        "stoi_min": df['stoi'].min(),
        "stoi_max": df['stoi'].max(),
        "processing_types": df['version_type'].value_counts().to_dict()
    }
    
    with open(os.path.join(Config.IMAGES_DIR, 'dataset_stats.txt'), 'w') as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    return df

if __name__ == "__main__":
    df = load_and_analyze_dataset()
    print("\nVisualizations saved to:", Config.IMAGES_DIR)
