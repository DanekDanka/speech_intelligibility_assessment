import os
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from scipy.signal import convolve

def add_reverberation(y, sr, ir_path=None, rt60=0.8, wet_level=0.3):
    """
    Добавляет реверберацию к аудиосигналу.
    
    Параметры:
        y: входной аудиосигнал
        sr: частота дискретизации
        ir_path: путь к импульсной характеристике (если None, генерируется искусственная)
        rt60: время реверберации в секундах (для искусственной реверберации)
        wet_level: уровень реверберации (0-1)
    """
    if ir_path is not None:
        # Загрузка импульсной характеристики из файла
        ir, sr_ir = librosa.load(ir_path, sr=sr)
        if sr_ir != sr:
            ir = librosa.resample(ir, orig_sr=sr_ir, target_sr=sr)
    else:
        # Генерация искусственной импульсной характеристики
        length = int(rt60 * sr)
        ir = np.random.normal(0, 1, length)
        ir = np.exp(-np.linspace(0, 10, length)) * ir
        ir = ir / np.max(np.abs(ir))
    
    # Применение свертки для добавления реверберации
    wet = convolve(y, ir, mode='same')
    
    # Смешивание оригинального и реверберированного сигнала
    dry_level = 1 - wet_level
    processed = dry_level * y + wet_level * wet
    
    # Нормализация
    processed = processed / np.max(np.abs(processed))
    
    return processed

def process_directory(input_dir, output_dir, ir_path=None, rt60=0.8, wet_level=0.3):
    """
    Обрабатывает все аудиофайлы в директории, добавляя реверберацию.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    audio_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.aiff'))]
    
    for filename in tqdm(audio_files, desc="Adding reverberation"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        y, sr = librosa.load(input_path, sr=None)
        
        # Добавление реверберации
        y_reverb = add_reverberation(y, sr, ir_path, rt60, wet_level)
        
        # Сохранение результата
        sf.write(output_path, y_reverb, sr)

if __name__ == "__main__":
    # Пути к файлам
    vad_directory = "/home/danya/develop/datasets/CMU-MOSEI/Audio/vad/"
    output_directory = "/home/danya/develop/datasets/CMU-MOSEI/Audio/vad_reverberation/"
    
    # Параметры реверберации
    ir_path = None  # Можно указать путь к файлу с импульсной характеристикой
    rt60 = 1.0      # Время реверберации в секундах
    wet_level = 0.5  # Уровень реверберации (0-1)
    
    process_directory(
        vad_directory,
        output_directory,
        ir_path=ir_path,
        rt60=rt60,
        wet_level=wet_level
    )
