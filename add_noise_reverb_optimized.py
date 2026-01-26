import os
import numpy as np
import soundfile as sf
import librosa
import random
from tqdm import tqdm
from scipy.signal import convolve
from pystoi import stoi
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings("ignore")

# Конфигурация
class Config:
    INPUT_DIR = "/home/danya/datasets/CMU-MOSEI/Audio/WAV_16000/"
    OUTPUT_DIR = "/home/danya/datasets/CMU-MOSEI/Audio/balanced_stoi_dataset/"
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
    NUM_WORKERS = 8  # Количество потоков для параллельной обработки
    CHUNK_SIZE = 4096  # Размер блока для обработки


def energy_based_vad(y, sr, top_db=20, frame_length=2048, hop_length=512):
    """Оптимизированная версия VAD с использованием numpy"""
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    energy = np.sum(frames**2, axis=0)
    energy_db = librosa.amplitude_to_db(energy, ref=np.max)
    speech_frames = energy_db > -top_db

    # Более эффективное расширение маски речи на семплы
    mask = np.zeros_like(y, dtype=bool)
    indices = np.where(speech_frames)[0]
    
    for i in indices:
        start = i * hop_length
        end = min(start + frame_length, len(y))
        mask[start:end] = True

    return y[mask]


def calculate_rms_db(y):
    """Расчет RMS в дБ"""
    rms = np.sqrt(np.mean(y**2))
    return librosa.amplitude_to_db([rms], ref=1.0)[0]


def add_noise_with_variable_snr(y, sr, target_snr):
    """Оптимизированная версия добавления шума с использованием numpy"""
    signal_rms = np.sqrt(np.mean(y**2))
    signal_db = librosa.amplitude_to_db([signal_rms], ref=1.0)[0]
    noise_db = signal_db - target_snr
    noise_level_db = librosa.db_to_amplitude(noise_db)
    
    # Более быстрое создание шума с использованием numpy
    noise = np.random.normal(0, float(noise_level_db), len(y))
    noisy_audio = y + noise
    
    # Более быстрое ограничение значений
    np.clip(noisy_audio, -1.0, 1.0, out=noisy_audio)
    return noisy_audio


# Кэш для импульсных откликов реверберации
_ir_cache = {}

def get_ir(sr, rt60):
    """Кэшированная генерация импульсного отклика"""
    key = (sr, rt60)
    if key not in _ir_cache:
        length = int(rt60 * sr)
        # Более быстрое создание IR
        t = np.linspace(0, 10, length)
        ir = np.random.normal(0, 1, length)
        ir = np.exp(-t) * ir
        ir = ir / (np.max(np.abs(ir)) + 1e-8)
        _ir_cache[key] = ir
    return _ir_cache[key]


def add_reverberation_optimized(y, sr, rt60=0.8, wet_level=0.3):
    """Оптимизированная версия с использованием более быстрого convolution"""
    ir = get_ir(sr, rt60)
    
    # Используем более быструю свёртку
    wet = convolve(y, ir, mode='same')
    
    # Нормализация с использованием numpy broadcasting
    dry_level = 1 - wet_level
    processed = dry_level * y + wet_level * wet
    
    max_val = np.max(np.abs(processed))
    if max_val > 1e-8:
        processed = processed / max_val
    
    return processed


def calculate_stoi_cached(clean, noisy, sr):
    """Версия вычисления STOI без изменений (уже оптимизирована в pystoi)"""
    min_len = min(len(clean), len(noisy))
    clean = clean[:min_len]
    noisy = noisy[:min_len]
    return stoi(clean, noisy, sr, extended=False)


def generate_audio_versions(y_vad, sr, file_counter, base_filename, output_dir):
    """Генерирует различные версии аудио без сохранения"""
    versions = []
    
    # 1. Оригинал после VAD (высокий STOI)
    versions.append(("vad_only", y_vad))
    
    # 2. Только шум (используем случайные SNR значения один раз)
    snr_noise_values = [random.uniform(snr_range[0], snr_range[1]) for snr_range in Config.SNR_RANGES]
    for i, target_snr in enumerate(snr_noise_values):
        noisy = add_noise_with_variable_snr(y_vad, sr, target_snr)
        versions.append((f"noise_snr_{target_snr:.1f}", noisy))
    
    # 3. Только реверберация
    for reverb in Config.REVERB_PARAMS:
        reverbed = add_reverberation_optimized(y_vad, sr, **reverb)
        versions.append((f"reverb_rt60_{reverb['rt60']}_wet_{reverb['wet_level']}", reverbed))
    
    # 4. Комбинации шума и реверберации (переиспользуем SNR значения)
    for snr_value in snr_noise_values:
        for reverb in Config.REVERB_PARAMS:
            noisy = add_noise_with_variable_snr(y_vad, sr, snr_value)
            noisy_reverbed = add_reverberation_optimized(noisy, sr, **reverb)
            versions.append((f"noise_snr_{snr_value:.1f}_reverb_rt60_{reverb['rt60']}_wet_{reverb['wet_level']}", 
                           noisy_reverbed))
    
    return versions


def save_and_calculate_stoi(args):
    """Функция для параллельной обработки (сохранение и вычисление STOI)"""
    try:
        version_name, processed_audio, y_vad, sr, output_filename, output_path, base_filename = args
        
        # Проверяем корректность аудио
        if processed_audio is None or len(processed_audio) == 0:
            return None
        
        # Сохраняем файл
        sf.write(output_path, processed_audio, sr)
        
        # Вычисляем STOI
        stoi_score = calculate_stoi_cached(y_vad, processed_audio, sr)
        
        # Проверяем корректность STOI
        if np.isnan(stoi_score) or np.isinf(stoi_score):
            return None
        
        return {
            "filename": output_filename,
            "version": version_name,
            "stoi": stoi_score,
            "original": base_filename + ".wav"
        }
    except Exception as e:
        print(f"  ✗ Ошибка при вычислении STOI для {version_name}: {e}")
        return None


def process_audio_file_optimized(input_path, output_dir, file_counter):
    """Оптимизированная обработка аудиофайла с параллельным сохранением"""
    try:
        y, sr = librosa.load(input_path, sr=None, mono=True)
        
        # Проверяем загрузку
        if y is None or len(y) == 0:
            print(f"  ✗ Не удалось загрузить {input_path}")
            return []
        
        # Применяем VAD
        y_vad = energy_based_vad(y, sr, **Config.VAD_PARAMS)
        
        # Проверяем результат VAD
        if y_vad is None or len(y_vad) == 0:
            print(f"  ✗ VAD вернул пустой результат для {input_path}")
            return []
        
        # Генерируем версии
        base_filename = os.path.splitext(os.path.basename(input_path))[0]
        versions = generate_audio_versions(y_vad, sr, file_counter, base_filename, output_dir)
        
        # Подготавливаем аргументы для параллельной обработки
        task_args = []
        for version_name, processed_audio in versions:
            output_filename = f"{base_filename}_{version_name}_{file_counter}.wav"
            output_path = os.path.join(output_dir, "audio", output_filename)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            task_args.append((version_name, processed_audio, y_vad, sr, output_filename, output_path, base_filename))
        
        # Обрабатываем параллельно с использованием ThreadPoolExecutor для I/O операций
        results = []
        with ThreadPoolExecutor(max_workers=min(4, len(task_args))) as executor:
            futures = [executor.submit(save_and_calculate_stoi, args) for args in task_args]
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
        
        return results
    except Exception as e:
        print(f"  ✗ Ошибка при обработке {input_path}: {e}")
        return []


def process_audio_files_batch(audio_files, output_dir):
    """Обрабатывает файлы с использованием пула потоков"""
    all_results = []
    
    def process_file_wrapper(args):
        filename, file_counter = args
        input_path = os.path.join(Config.INPUT_DIR, filename)
        try:
            results = process_audio_file_optimized(input_path, output_dir, file_counter)
            return results
        except Exception as e:
            print(f"Ошибка при обработке {filename}: {e}")
            return []
    
    # Используем пул потоков для параллельной обработки файлов
    with ThreadPoolExecutor(max_workers=Config.NUM_WORKERS) as executor:
        futures = [
            executor.submit(process_file_wrapper, (filename, idx))
            for idx, filename in enumerate(audio_files)
        ]
        
        # Собираем результаты с progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            results = future.result()
            all_results.extend(results)
    
    return all_results


def create_balanced_dataset_optimized():
    """Оптимизированная версия создания датасета"""
    # Создаем выходные директории
    os.makedirs(os.path.join(Config.OUTPUT_DIR, "audio"), exist_ok=True)
    csv_path = os.path.join(Config.OUTPUT_DIR, "metadata.csv")
    
    # Получаем список аудиофайлов
    audio_files = [f for f in os.listdir(Config.INPUT_DIR) 
                 if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.aiff'))]
    
    print(f"Найдено {len(audio_files)} аудиофайлов для обработки")
    
    # Обрабатываем файлы параллельно
    all_results = process_audio_files_batch(audio_files, Config.OUTPUT_DIR)
    
    # Фильтруем результаты (удаляем None)
    all_results = [r for r in all_results if r is not None]
    
    if len(all_results) == 0:
        print("Ошибка: нет успешно обработанных результатов!")
        return None
    
    # Создаем DataFrame с метаданными
    df = pd.DataFrame(all_results)
    
    # Проверяем наличие колонки 'stoi'
    if 'stoi' not in df.columns:
        print("Ошибка: колонка 'stoi' не найдена в результатах!")
        return None
    
    # Балансируем датасет по STOI бинам
    balanced_files = []
    bin_edges = np.linspace(0, 1, Config.STOI_BINS + 1)
    
    print("Балансирование датасета...")
    for i in range(Config.STOI_BINS):
        lower = bin_edges[i]
        upper = bin_edges[i+1]
        
        # Выбираем файлы из текущего бина
        if lower == 1.0 and upper == 1.0:  # Последний бин
            bin_files = df[df['stoi'] >= lower]
        else:
            bin_files = df[(df['stoi'] >= lower) & (df['stoi'] < upper)]
        
        # Если файлов недостаточно, используем повторения
        if len(bin_files) > 0:
            if len(bin_files) < Config.TARGET_SAMPLES_PER_BIN:
                repeats = (Config.TARGET_SAMPLES_PER_BIN // len(bin_files)) + 1
                bin_files = pd.concat([bin_files]*repeats, ignore_index=True)
            
            # Выбираем нужное количество случайных файлов
            balanced_files.append(bin_files.sample(n=Config.TARGET_SAMPLES_PER_BIN, replace=True, random_state=42))
        else:
            print(f"Предупреждение: бин [{lower:.1f}, {upper:.1f}) пуст")
    
    # Объединяем сбалансированные данные
    balanced_df = pd.concat(balanced_files, ignore_index=True)
    
    # Сохраняем метаданные
    balanced_df.to_csv(csv_path, index=False)
    
    print(f"\nСоздан сбалансированный датасет с {len(balanced_df)} файлами.")
    print(f"Распределение STOI:")
    print(balanced_df['stoi'].describe())
    
    # Визуализация распределения STOI
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.hist(balanced_df['stoi'], bins=Config.STOI_BINS)
        plt.title("STOI Distribution in Balanced Dataset")
        plt.xlabel("STOI score")
        plt.ylabel("Number of samples")
        plt.savefig(os.path.join(Config.OUTPUT_DIR, "stoi_distribution.png"))
        plt.close()
    except ImportError:
        print("matplotlib не установлена, пропускаем визуализацию")
    
    return balanced_df


if __name__ == "__main__":
    df = create_balanced_dataset_optimized()
    print(f"\nОптимизация завершена!")
    print(f"Результаты сохранены в: {Config.OUTPUT_DIR}")

