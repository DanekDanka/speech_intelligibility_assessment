import numpy as np
import librosa


def get_vad_mask(y, sr, top_db=20, frame_length=2048, hop_length=512):
    """
    Получает маску VAD для определения речевых и неречевых сегментов.
    
    Args:
        y: аудио сигнал
        sr: частота дискретизации
        top_db: порог в dB для определения речи
        frame_length: длина фрейма
        hop_length: шаг между фреймами
    
    Returns:
        Маска (True для речи, False для неречи)
    """
    # Убеждаемся, что все параметры - это числа
    top_db = float(top_db)
    frame_length = int(frame_length)
    hop_length = int(hop_length)
    
    # Убеждаемся, что y - это numpy array и 1D
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.flatten()
    
    # Используем более надежную реализацию фрейминга
    # Вычисляем количество фреймов
    n_frames = int((len(y) - frame_length) / hop_length) + 1
    if n_frames <= 0:
        # Если сигнал слишком короткий, считаем весь сигнал речью
        return np.ones(len(y), dtype=bool)
    
    # Создаем массив фреймов
    frames = np.zeros((frame_length, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        if end <= len(y):
            frames[:, i] = y[start:end]
        else:
            # Последний фрейм может быть неполным
            frames[:len(y)-start, i] = y[start:]
    
    # Вычисляем энергию каждого фрейма
    energy = np.sum(frames**2, axis=0)
    # Убеждаемся, что energy - это numpy array
    energy = np.asarray(energy, dtype=np.float32)
    # Убеждаемся, что energy - это 1D массив
    if energy.ndim == 0:
        energy = np.array([energy], dtype=np.float32)
    elif energy.ndim > 1:
        energy = energy.flatten()
    # Защита от деления на ноль
    if len(energy) == 0:
        energy = np.ones(max(1, n_frames), dtype=np.float32) * np.float32(1e-10)
    elif np.max(energy) == 0:
        energy = np.ones_like(energy, dtype=np.float32) * np.float32(1e-10)
    # Вычисляем максимальное значение для ref
    ref_value = float(np.max(energy))
    if ref_value <= 0:
        ref_value = 1.0
    # Убеждаемся, что energy - это numpy array перед вызовом librosa
    energy = np.asarray(energy, dtype=np.float32)
    energy_db = librosa.amplitude_to_db(energy, ref=ref_value)
    # Убеждаемся, что energy_db - это numpy array
    energy_db = np.asarray(energy_db, dtype=np.float32)
    # Убеждаемся, что energy_db - это 1D массив
    if energy_db.ndim == 0:
        energy_db = np.array([energy_db], dtype=np.float32)
    elif energy_db.ndim > 1:
        energy_db = energy_db.flatten()
    # Убеждаемся, что top_db - это число
    top_db = float(top_db)
    speech_frames = energy_db > -top_db
    # Убеждаемся, что speech_frames - это numpy array
    speech_frames = np.asarray(speech_frames, dtype=bool)

    mask = np.zeros(len(y), dtype=bool)
    for i, is_speech in enumerate(speech_frames):
        if is_speech:
            start = i * hop_length
            end = min(start + frame_length, len(y))
            mask[start:end] = True

    return mask

