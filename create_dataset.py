import tensorflow as tf
import os
import numpy as np
from scipy.io import wavfile
import uuid
import librosa


def load_wav_file(file_path):
    # Загрузка .wav файла и извлечение отсчётов
    samples, sample_rate = librosa.load(file_path, sr=None)
    return samples, sample_rate

def add_noise(samples, noise_level=0.1):
    # Генерация шума
    noise = np.random.normal(0, noise_level, samples.shape)
    # Добавление шума к сигналу
    noisy_samples = samples + noise
    # Вычисление мощности шума
    power = np.mean(noise ** 2)
    print('Noise power:', power)
    print('Signal power:', np.mean(samples ** 2))
    return noisy_samples, power

def wav_generator(directory):
    # Получение списка всех .wav файлов в папке
    wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]

    for wav_file in wav_files:
        file_path = os.path.join(directory, wav_file)
        samples, sample_rate = load_wav_file(file_path)
        noisy_samples, power = add_noise(samples)
        yield noisy_samples, power

def create_dataset(directory):
    # Создание TensorFlow Dataset из генератора
    dataset = tf.data.Dataset.from_generator(
        lambda: wav_generator(directory),
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32)
        )
    )
    return dataset


if __name__ == '__main__':
    dir_cut_data = 'cut\\'
    dataset = create_dataset(dir_cut_data)
    dataset.save('dataset')

