import numpy as np

from vad import get_vad_mask


def calculate_power(y):
    """
    Вычисляет мощность сигнала.
    
    Args:
        y: аудио сигнал
    
    Returns:
        Мощность (средний квадрат амплитуды)
    """
    # Убеждаемся, что y - это numpy array
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0:
        return 0.0
    return float(np.mean(y**2))


def calculate_snr_from_powers(speech_power, noise_power):
    """
    Вычисляет SNR в dB из мощностей.
    
    Args:
        speech_power: мощность речи
        noise_power: мощность шума
    
    Returns:
        SNR в dB
    """
    if noise_power == 0:
        return np.inf
    snr_linear = speech_power / noise_power
    snr_db = 10 * np.log10(snr_linear)
    return snr_db


def add_white_noise_with_target_snr(y, sr, target_snr_db, speech_mask):
    """
    Добавляет белый шум к аудио для достижения целевого SNR.
    
    SNR вычисляется как отношение мощности речи к мощности добавленного шума.
    
    Args:
        y: исходный аудио сигнал
        sr: частота дискретизации
        target_snr_db: целевой SNR в dB
        speech_mask: маска VAD (True для речи, False для неречи)
    
    Returns:
        Зашумленный аудио сигнал
    """
    # Убеждаемся, что y - это numpy array и 1D
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.flatten()
    
    # Убеждаемся, что speech_mask - это numpy array
    speech_mask = np.asarray(speech_mask, dtype=bool)
    
    # Вычисляем мощность речи
    speech_segments = y[speech_mask]
    if len(speech_segments) == 0:
        # Если речи нет, просто добавляем шум с фиксированным уровнем
        noise_power = 0.01  # Небольшой уровень шума по умолчанию
    else:
        speech_power = calculate_power(speech_segments)
        
        # Вычисляем требуемую мощность шума для достижения целевого SNR
        # SNR = 10*log10(P_speech / P_noise)
        # P_noise = P_speech / 10^(SNR/10)
        noise_power = float(speech_power) / (10 ** (float(target_snr_db) / 10.0))
    
    # Генерируем белый шум с нужной мощностью
    # Для белого шума: std = sqrt(power)
    noise_std = float(np.sqrt(noise_power))
    signal_length = int(len(y))
    noise = np.random.normal(0.0, noise_std, signal_length).astype(np.float32)
    
    # Добавляем шум к сигналу
    noisy_audio = y + noise
    
    # Ограничиваем значения в диапазоне [-1, 1]
    noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
    
    return noisy_audio


def get_target_snr_for_uniform_distribution(distribution, snr_min, snr_max, n_bins):
    """
    Выбирает целевой SNR из бина с наименьшим количеством файлов для равномерного распределения.
    Приоритет отдается бинам с низким SNR для получения большего количества неразборчивых записей.
    
    Args:
        distribution: словарь с количеством файлов в каждом бине
        snr_min: минимальное значение SNR
        snr_max: максимальное значение SNR
        n_bins: количество бинов
    
    Returns:
        Целевой SNR в dB и номер бина
    """
    # Убеждаемся, что все параметры - это числа
    snr_min = float(snr_min)
    snr_max = float(snr_max)
    n_bins = int(n_bins)
    
    bin_width = (snr_max - snr_min) / n_bins
    
    # Создаем список всех бинов с их количеством файлов (0 для пустых)
    bin_counts = []
    for bin_idx in range(n_bins):
        bin_key = str(bin_idx)
        count = distribution.get(bin_key, 0)
        bin_counts.append((bin_idx, count))
    
    # Находим минимальное количество файлов
    min_count = min(count for _, count in bin_counts)
    
    # Находим все бины с минимальным количеством
    min_bins = [bin_idx for bin_idx, count in bin_counts if count == min_count]
    
    # Если есть несколько бинов с минимальным количеством, приоритет отдаем низким SNR
    # Сортируем по индексу бина (меньший индекс = меньший SNR)
    min_bins.sort()
    
    # Если минимальное количество = 0 (есть пустые бины), приоритет отдаем пустым бинам
    # и среди них выбираем низкие SNR
    if min_count == 0:
        # Выбираем из пустых бинов, приоритет низким SNR
        # С вероятностью 90% выбираем из первых 40% пустых бинов (самые низкие SNR)
        low_snr_threshold = max(1, int(len(min_bins) * 0.4))
        if np.random.random() < 0.9 and low_snr_threshold > 0:
            selected_bin = int(np.random.choice(min_bins[:low_snr_threshold]))
        else:
            selected_bin = int(np.random.choice(min_bins))
    elif len(min_bins) > 1:
        # Если все бины заполнены, выбираем из минимальных с приоритетом низким SNR
        # С вероятностью 70% выбираем из первых 50% (низкие SNR)
        low_snr_threshold = max(1, int(len(min_bins) * 0.5))
        if np.random.random() < 0.7 and low_snr_threshold > 0:
            selected_bin = int(np.random.choice(min_bins[:low_snr_threshold]))
        else:
            selected_bin = int(np.random.choice(min_bins))
    else:
        selected_bin = int(min_bins[0]) if min_bins else np.random.randint(0, n_bins)
    
    # Убеждаемся, что selected_bin - это int
    selected_bin = int(selected_bin)
    
    # Выбираем случайное значение SNR внутри бина
    bin_start = snr_min + selected_bin * bin_width
    bin_end = bin_start + bin_width
    target_snr = np.random.uniform(bin_start, bin_end)
    
    return target_snr, selected_bin


def get_snr_bin(snr_value, snr_min, snr_max, n_bins):
    """
    Определяет номер бина для значения SNR.
    
    Args:
        snr_value: значение SNR в dB
        snr_min: минимальное значение SNR
        snr_max: максимальное значение SNR
        n_bins: количество бинов
    
    Returns:
        Номер бина
    """
    if snr_value < snr_min:
        return 0
    if snr_value >= snr_max:
        return n_bins - 1
    bin_width = (snr_max - snr_min) / n_bins
    bin_idx = int((snr_value - snr_min) / bin_width)
    return min(bin_idx, n_bins - 1)

