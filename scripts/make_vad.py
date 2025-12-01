import os
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from pathlib import Path

# Конфигурация
INPUT_DIR = "/home/danya/datasets/CMU-MOSEI/Audio/WAV_16000/"
OUTPUT_DIR = "/home/danya/datasets/CMU-MOSEI/Audio/VAD/"

# Параметры VAD
VAD_PARAMS = {
    "top_db": 20,
    "frame_length": 2048,
    "hop_length": 512
}

def energy_based_vad(y, sr, top_db=20, frame_length=2048, hop_length=512):
    """
    Применяет VAD на основе энергии сигнала.
    
    Args:
        y: аудио сигнал
        sr: частота дискретизации
        top_db: порог в dB для определения речи
        frame_length: длина фрейма
        hop_length: шаг между фреймами
    
    Returns:
        Отфильтрованный аудио сигнал (только сегменты с речью)
    """
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

def process_wav_file(input_path, output_path):
    """
    Обрабатывает один wav файл через VAD и сохраняет результат.
    
    Args:
        input_path: путь к входному файлу
        output_path: путь для сохранения результата
    """
    try:
        # Загружаем аудио
        y, sr = librosa.load(input_path, sr=None, mono=True)
        
        # Применяем VAD
        y_vad = energy_based_vad(y, sr, **VAD_PARAMS)
        
        # Если после VAD ничего не осталось, пропускаем файл
        if len(y_vad) == 0:
            print(f"Warning: No speech detected in {input_path}, skipping...")
            return False
        
        # Сохраняем результат
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, y_vad, sr)
        
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False

def process_dataset():
    """
    Обрабатывает все wav файлы из входной директории через VAD.
    """
    input_path = Path(INPUT_DIR)
    output_path = Path(OUTPUT_DIR)
    
    # Находим все wav файлы (рекурсивно)
    wav_files = list(input_path.rglob("*.wav"))
    
    if len(wav_files) == 0:
        print(f"No wav files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(wav_files)} wav files to process")
    
    # Обрабатываем каждый файл
    success_count = 0
    failed_count = 0
    
    for wav_file in tqdm(wav_files, desc="Processing files"):
        # Вычисляем относительный путь от входной директории
        relative_path = wav_file.relative_to(input_path)
        
        # Формируем путь для выходного файла
        output_file = output_path / relative_path
        
        # Обрабатываем файл
        if process_wav_file(str(wav_file), str(output_file)):
            success_count += 1
        else:
            failed_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_dataset()

