#!/usr/bin/env python3
"""
Проверка файлов Рины: моно или стерео.
Ищет все .wav в указанной папке, в имени которых есть "Рина",
и выводит для каждого число каналов (1 = моно, 2 = стерео).
"""

import os
import sys
from pathlib import Path
import wave

from scipy.io import wavfile


def check_channels(wav_path: str) -> tuple[int, str]:
    """
    Читает WAV и возвращает (число_каналов, "моно"|"стерео"|"N каналов").
    """
    sr, data = wavfile.read(wav_path)
    if data.ndim == 1:
        return 1, "моно"
    channels = data.shape[1]
    print("data.shape")
    if channels == 2:
        return 2, "стерео"
    return channels, f"{channels} каналов"

def check_wav_channels(file_path):
    with wave.open(file_path, 'rb') as wf:
        nchannels = wf.getnchannels()
        if nchannels == 1:
            return "Моно"
        elif nchannels == 2:
            return "Стерео"
        else:
            return f"{nchannels} каналов (многоканальный)"

def main():
    # Папка по умолчанию — как в make_dataset.ipynb
    default_folder = "/home/danya/datasets/speech_thesisis/"
    folder = sys.argv[1] if len(sys.argv) > 1 else default_folder
    folder = Path(folder)
    if not folder.is_dir():
        print(f"Ошибка: папка не найдена: {folder}")
        sys.exit(1)

    wav_files = sorted(f for f in folder.glob("*.wav") if "Рина" in f.name)
    if not wav_files:
        print(f"В папке {folder} не найдено WAV-файлов с 'Рина' в имени.")
        sys.exit(0)

    print(f"Папка: {folder}")
    print(f"Найдено файлов с 'Рина': {len(wav_files)}\n")
    print(f"{'Файл':<45} {'Каналы':<12} {'Тип'}")
    print("-" * 65)

    for path in wav_files:
        try:
            n_channels, label = check_channels(str(path))
            print(f"{path.name:<45} {n_channels:<12} {label}")
        except Exception as e:
            print(f"{path.name:<45} ошибка: {e}")

    print("-" * 65)


if __name__ == "__main__":
    main()
