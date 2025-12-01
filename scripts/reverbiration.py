import numpy as np
from scipy.signal import convolve

from vad import get_vad_mask


def add_reverberation(y, sr, rt60=0.8, wet_level=0.3):
    """
    Добавляет реверберацию к аудио сигналу.
    
    Args:
        y: исходный аудио сигнал
        sr: частота дискретизации
        rt60: время реверберации в секундах (время затухания на 60 dB)
        wet_level: уровень реверберации (0-1), где 0 - только сухой сигнал, 1 - только реверберация
    
    Returns:
        Аудио сигнал с реверберацией
    """
    # Убеждаемся, что y - это numpy array и 1D
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.flatten()
    
    # Убеждаемся, что параметры - это числа
    rt60 = float(rt60)
    wet_level = float(wet_level)
    sr = float(sr)
    
    # Создаем импульсную характеристику (impulse response)
    # Длина импульсной характеристики должна соответствовать RT60
    length = int(rt60 * sr)
    
    # Генерируем случайную импульсную характеристику
    # Используем экспоненциальное затухание для моделирования реверберации
    ir = np.random.normal(0, 1, length).astype(np.float32)
    
    # Применяем экспоненциальное затухание
    # RT60 означает, что сигнал затухает на 60 dB за время rt60
    # Экспоненциальное затухание: exp(-t / tau), где tau = rt60 / ln(1000)
    decay = np.exp(-np.linspace(0, 10, length))
    ir = ir * decay
    
    # Нормализуем импульсную характеристику
    max_ir = np.max(np.abs(ir))
    if max_ir > 0:
        ir = ir / max_ir
    
    # Применяем свертку для добавления реверберации
    wet = convolve(y, ir, mode='same')
    
    # Смешиваем сухой и мокрый сигналы
    dry_level = 1.0 - wet_level
    processed = dry_level * y + wet_level * wet
    
    # Нормализуем, чтобы избежать клиппинга
    max_val = np.max(np.abs(processed))
    if max_val > 0:
        processed = processed / max_val
    
    # Ограничиваем значения в диапазоне [-1, 1]
    processed = np.clip(processed, -1.0, 1.0)
    
    return processed


def get_target_rt60_for_uniform_distribution(distribution, rt60_min, rt60_max, n_bins):
    """
    Выбирает целевой RT60 из бина с наименьшим количеством файлов для равномерного распределения.
    Приоритет отдается бинам с высоким RT60 для получения большего количества записей с сильной реверберацией.
    
    Args:
        distribution: словарь с количеством файлов в каждом бине
        rt60_min: минимальное значение RT60
        rt60_max: максимальное значение RT60
        n_bins: количество бинов
    
    Returns:
        Целевой RT60 в секундах и номер бина
    """
    # Убеждаемся, что все параметры - это числа
    rt60_min = float(rt60_min)
    rt60_max = float(rt60_max)
    n_bins = int(n_bins)
    
    bin_width = (rt60_max - rt60_min) / n_bins
    
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
    
    # Если есть несколько бинов с минимальным количеством, приоритет отдаем высоким RT60
    # Сортируем по индексу бина в обратном порядке (больший индекс = больший RT60)
    min_bins.sort(reverse=True)
    
    # Если минимальное количество = 0 (есть пустые бины), приоритет отдаем пустым бинам
    # и среди них выбираем высокие RT60
    if min_count == 0:
        # Выбираем из пустых бинов, приоритет высоким RT60
        # С вероятностью 90% выбираем из последних 40% пустых бинов (самые высокие RT60)
        high_rt60_threshold = max(1, int(len(min_bins) * 0.4))
        if np.random.random() < 0.9 and high_rt60_threshold > 0:
            selected_bin = int(np.random.choice(min_bins[:high_rt60_threshold]))
        else:
            selected_bin = int(np.random.choice(min_bins))
    elif len(min_bins) > 1:
        # Если все бины заполнены, выбираем из минимальных с приоритетом высоким RT60
        # С вероятностью 70% выбираем из последних 50% (высокие RT60)
        high_rt60_threshold = max(1, int(len(min_bins) * 0.5))
        if np.random.random() < 0.7 and high_rt60_threshold > 0:
            selected_bin = int(np.random.choice(min_bins[:high_rt60_threshold]))
        else:
            selected_bin = int(np.random.choice(min_bins))
    else:
        selected_bin = int(min_bins[0]) if min_bins else np.random.randint(0, n_bins)
    
    # Убеждаемся, что selected_bin - это int
    selected_bin = int(selected_bin)
    
    # Выбираем случайное значение RT60 внутри бина
    bin_start = rt60_min + selected_bin * bin_width
    bin_end = bin_start + bin_width
    target_rt60 = np.random.uniform(bin_start, bin_end)
    
    return target_rt60, selected_bin


def get_rt60_bin(rt60_value, rt60_min, rt60_max, n_bins):
    """
    Определяет номер бина для значения RT60.
    
    Args:
        rt60_value: значение RT60 в секундах
        rt60_min: минимальное значение RT60
        rt60_max: максимальное значение RT60
        n_bins: количество бинов
    
    Returns:
        Номер бина
    """
    if rt60_value < rt60_min:
        return 0
    if rt60_value >= rt60_max:
        return n_bins - 1
    bin_width = (rt60_max - rt60_min) / n_bins
    bin_idx = int((rt60_value - rt60_min) / bin_width)
    return min(bin_idx, n_bins - 1)

