import numpy as np
import sounddevice as sd
import librosa
import soundfile as sf
import tensorflow as tf
import os
import tempfile
from tensorflow.keras import models, losses, metrics
from pystoi import stoi
from scipy.signal import resample

# Загрузка обученной модели
try:
    model = models.load_model(
        '/home/danya/develop/models/stoi_predictor.h5',
        custom_objects={
            'mse': losses.MeanSquaredError(),
            'mae': metrics.MeanAbsoluteError(),
            'MeanAbsoluteError': metrics.MeanAbsoluteError()
        }
    )
except:
    def build_model(input_shape):
        model = models.Sequential([
            tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling1D(2),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    model = build_model((500, 20))
    model.load_weights('/home/danya/develop/models/stoi_predictor.h5')

# Параметры
SAMPLE_RATE = 48000
TARGET_SR = 48000  # Для VAD и pystoi
DURATION = 5  # Увеличим длительность записи
N_MFCC = 20
MAX_FRAMES = 500
VAD_PARAMS = {
    'top_db': 25,  # Более мягкий порог
    'frame_length': 2048,
    'hop_length': 512
}

def energy_based_vad(y, sr, top_db=20, frame_length=2048, hop_length=512):
    """VAD с возвратом только речевых сегментов"""
    # Разбиваем на фреймы
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    
    # Вычисляем энергию
    energy = np.sum(frames**2, axis=0)
    energy_db = librosa.amplitude_to_db(energy, ref=np.max)
    
    # Определяем речевые фреймы
    speech_frames = energy_db > -top_db
    
    # Собираем только речевые фреймы
    speech_audio = []
    for i, is_speech in enumerate(speech_frames):
        if is_speech:
            start = i * hop_length
            end = start + frame_length
            speech_audio.extend(y[start:end])
    
    return np.array(speech_audio)

def resample_audio(audio, original_sr, target_sr):
    """Ресемплирование аудио с сохранением качества"""
    duration = len(audio) / original_sr
    num_samples = int(duration * target_sr)
    return resample(audio, num_samples)

def record_audio():
    """Запись аудио с микрофона"""
    print(f"Говорите в микрофон в течение {DURATION} секунд...")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                  samplerate=SAMPLE_RATE,
                  channels=1,
                  dtype='float32')
    sd.wait()
    return audio.flatten()

def process_audio(audio):
    """Полная обработка аудио: VAD и подготовка для анализа"""
    # Ресемплируем для VAD
    audio_16k = resample_audio(audio, SAMPLE_RATE, TARGET_SR)
    
    # Применяем VAD и получаем только речь
    speech_only = energy_based_vad(audio_16k, TARGET_SR, **VAD_PARAMS)
    
    # Если нет речи - возвращаем оригинал с предупреждением
    if len(speech_only) == 0:
        print("Предупреждение: VAD не обнаружил речевых сегментов!")
        speech_only = audio_16k  # Возвращаем оригинал
    
    # Ресемплируем обратно для модели
    speech_for_model = resample_audio(speech_only, TARGET_SR, SAMPLE_RATE)
    
    return {
        'original': audio,
        'speech_only': speech_only,
        'speech_for_model': speech_for_model,
        'sr_vad': TARGET_SR,
        'sr_model': SAMPLE_RATE
    }

def extract_mfcc(audio, sr=SAMPLE_RATE):
    """Извлечение MFCC с нормализацией"""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    
    # Приведение к фиксированной длине
    if mfcc.shape[1] < MAX_FRAMES:
        pad_width = ((0, 0), (0, MAX_FRAMES - mfcc.shape[1]))
        mfcc = np.pad(mfcc, pad_width, mode='constant')
    else:
        mfcc = mfcc[:, :MAX_FRAMES]
    
    return mfcc.T

def calculate_pystoi(clean, processed, sr):
    """Точный расчет STOI с помощью pystoi"""
    min_len = min(len(clean), len(processed))
    return stoi(clean[:min_len], processed[:min_len], sr, extended=False)

def predict_stoi(audio, sr=SAMPLE_RATE):
    """Предсказание STOI моделью"""
    mfcc = extract_mfcc(audio, sr)
    return model.predict(np.expand_dims(mfcc, axis=0), verbose=0)[0][0]

def save_audio_segments(audio_dict):
    """Сохранение аудио сегментов для анализа"""
    temp_dir = tempfile.gettempdir()
    paths = {}
    
    # Оригинальное аудио
    paths['original'] = os.path.join(temp_dir, "original.wav")
    sf.write(paths['original'], audio_dict['original'], audio_dict['sr_model'])
    
    # Только речь (после VAD)
    paths['speech_only'] = os.path.join(temp_dir, "speech_only.wav")
    sf.write(paths['speech_only'], audio_dict['speech_only'], audio_dict['sr_vad'])
    
    return paths

def main():
    print("Доступные аудио устройства:")
    print(sd.query_devices())
    
    sd.default.device = None
    sd.default.samplerate = SAMPLE_RATE

    while True:
        input("\nНажмите Enter чтобы начать запись...")

        try:
            # 1. Запись аудио
            original_audio = record_audio()
            
            # 2. Обработка VAD
            processed = process_audio(original_audio)
            
            # 3. Сохранение для проверки
            audio_paths = save_audio_segments(processed)
            print(f"\nОригинальная запись сохранена: {audio_paths['original']}")
            print(f"Речь после VAD сохранена: {audio_paths['speech_only']}")
            
            # 4. Расчет STOI разными методами
            # Для оригинального аудио (без VAD)
            orig_stoi_model = predict_stoi(original_audio)
            
            # Для очищенного аудио (после VAD)
            vad_stoi_model = predict_stoi(processed['speech_for_model'])
            
            # Расчет pystoi для сравнения
            clean_16k = resample_audio(original_audio, SAMPLE_RATE, TARGET_SR)
            vad_stoi_pystoi = calculate_pystoi(clean_16k, processed['speech_only'], TARGET_SR)
            
            # 5. Вывод результатов
            print("\n=== Результаты анализа ===")
            print(f"STOI после VAD (модель): {vad_stoi_model:.4f}")
            print(f"STOI после VAD (pystoi): {vad_stoi_pystoi:.4f}")
            
            # 6. Интерпретация
            print("\nИнтерпретация качества после VAD:")
            if vad_stoi_model > 0.75:
                print("Отличное качество речи!")
            elif vad_stoi_model > 0.55:
                print("Хорошее качество речи")
            elif vad_stoi_model > 0.35:
                print("Удовлетворительное качество")
            else:
                print("Низкое качество речи")

        except Exception as e:
            print(f"\nОшибка: {str(e)}")

        if input("\nПродолжить? (y/n): ").lower() != 'y':
            print("Завершение работы...")
            break

if __name__ == "__main__":
    main()
