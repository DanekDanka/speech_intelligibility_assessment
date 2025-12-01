"""
Dataset для загрузки аудио файлов и предсказания STOI
"""
import os
import re
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import librosa


class STOIDataset(Dataset):
    """
    Dataset для предсказания STOI из аудио файлов.
    
    Поддерживает файлы с форматом имени:
    - snr=12_34__name=original_filename.wav (только шум)
    - rt60=0_87__wet=0_5__name=original_filename.wav (только реверберация)
    - snr=12_34__rt60=0_87__wet=0_5__name=original_filename.wav (шум + реверберация)
    """
    
    def __init__(self, 
                 audio_dir,
                 original_dir,
                 sample_rate=16000,
                 max_length_seconds=10,
                 use_wav2vec=True,
                 target_stoi=None):
        """
        Args:
            audio_dir: Директория с обработанными аудио файлами
            original_dir: Директория с оригинальными аудио файлами (для вычисления STOI)
            sample_rate: Частота дискретизации
            max_length_seconds: Максимальная длина аудио в секундах
            use_wav2vec: Использовать ли wav2vec для извлечения признаков
            target_stoi: Если указан, используется как целевое значение STOI (для тестирования)
        """
        self.audio_dir = Path(audio_dir)
        self.original_dir = Path(original_dir)
        self.sample_rate = sample_rate
        self.max_length = int(max_length_seconds * sample_rate)
        self.use_wav2vec = use_wav2vec
        
        # Проверяем существование директорий
        if not self.audio_dir.exists():
            raise ValueError(f"Директория с аудио файлами не найдена: {self.audio_dir}")
        if not self.original_dir.exists():
            raise ValueError(f"Директория с оригинальными файлами не найдена: {self.original_dir}")
        
        # Находим все аудио файлы
        self.audio_files = list(self.audio_dir.rglob("*.wav"))
        print(f"Найдено {len(self.audio_files)} аудио файлов в {self.audio_dir}")
        
        if len(self.audio_files) == 0:
            print(f"ВНИМАНИЕ: Не найдено ни одного .wav файла в {self.audio_dir}")
            print(f"Проверьте путь к директории с обработанными аудио файлами")
            self.valid_files = []
            return
        
        # Фильтруем файлы, для которых есть оригиналы
        self.valid_files = []
        not_found_count = 0
        sample_filenames = []
        
        for audio_file in self.audio_files:
            original_file = self._find_original_file(audio_file)
            if original_file is not None:
                self.valid_files.append((audio_file, original_file))
            else:
                not_found_count += 1
                if len(sample_filenames) < 5:  # Сохраняем первые 5 примеров
                    original_filename = self._extract_original_filename(audio_file.name)
                    sample_filenames.append((audio_file.name, original_filename))
        
        print(f"Найдено {len(self.valid_files)} валидных файлов")
        if not_found_count > 0:
            print(f"Не найдено оригиналов для {not_found_count} файлов")
            if sample_filenames:
                print("Примеры файлов без оригиналов:")
                for proc_name, orig_name in sample_filenames[:3]:
                    print(f"  Обработанный: {proc_name}")
                    print(f"  Ищется оригинал: {orig_name}")
                    # Проверяем, существует ли файл с таким именем
                    test_path = self.original_dir / orig_name
                    print(f"  Путь: {test_path} (существует: {test_path.exists()})")
                    # Пробуем найти рекурсивно
                    found = list(self.original_dir.rglob(orig_name))
                    print(f"  Найдено рекурсивно: {len(found)} файлов")
                    if found:
                        print(f"  Первый найденный: {found[0]}")
        
        # Если target_stoi указан, используем его вместо вычисления
        self.target_stoi = target_stoi
        
    def _find_original_file(self, processed_file):
        """Находит оригинальный файл по имени обработанного файла"""
        filename = processed_file.name
        original_filename = self._extract_original_filename(filename)
        
        if not original_filename:
            return None
        
        # Сначала пробуем прямой путь
        original_path = self.original_dir / original_filename
        if original_path.exists():
            return original_path
        
        # Пробуем найти рекурсивно (с учетом возможных поддиректорий)
        found_files = list(self.original_dir.rglob(original_filename))
        if found_files:
            return found_files[0]
        
        # Если не нашли, пробуем найти по имени без расширения
        original_name_no_ext = Path(original_filename).stem
        found_files = list(self.original_dir.rglob(f"{original_name_no_ext}.wav"))
        if found_files:
            return found_files[0]
        
        return None
    
    def _extract_original_filename(self, filename):
        """Извлекает оригинальное имя файла из имени обработанного файла"""
        match = re.search(r'__name=(.+)$', filename)
        if match:
            original_name = match.group(1)
            if not original_name.endswith('.wav'):
                return original_name + '.wav'
            return original_name
        if '__' in filename:
            return filename.split('__')[-1]
        return filename
    
    def _parse_filename(self, filename):
        """Парсит параметры из имени файла"""
        params = {}
        
        # SNR
        snr_match = re.search(r'^snr=([0-9_\-]+)__', filename)
        if snr_match:
            snr_str = snr_match.group(1).replace('_', '.')
            try:
                params['snr'] = float(snr_str)
            except:
                params['snr'] = None
        else:
            params['snr'] = None
        
        # RT60
        rt60_match = re.search(r'__rt60=([0-9_]+)__', filename)
        if not rt60_match:
            rt60_match = re.search(r'^rt60=([0-9_]+)__', filename)
        if rt60_match:
            rt60_str = rt60_match.group(1).replace('_', '.')
            try:
                params['rt60'] = float(rt60_str)
            except:
                params['rt60'] = None
        else:
            params['rt60'] = None
        
        # Wet level
        wet_match = re.search(r'__wet=([0-9_]+)__', filename)
        if wet_match:
            wet_str = wet_match.group(1).replace('_', '.')
            try:
                params['wet'] = float(wet_str)
            except:
                params['wet'] = None
        else:
            params['wet'] = None
        
        return params
    
    def _load_audio(self, filepath):
        """Загружает аудио файл"""
        try:
            waveform, sr = torchaudio.load(str(filepath))
            # Конвертируем в моно
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Ресемплируем если нужно
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            
            # Обрезаем или дополняем до нужной длины
            if waveform.shape[1] > self.max_length:
                waveform = waveform[:, :self.max_length]
            elif waveform.shape[1] < self.max_length:
                padding = self.max_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            
            return waveform.squeeze(0)  # Убираем dimension каналов
        except Exception as e:
            print(f"Ошибка загрузки {filepath}: {e}")
            return torch.zeros(self.max_length)
    
    def _calculate_stoi(self, processed_file, original_file):
        """Вычисляет STOI между обработанным и оригинальным файлом"""
        if self.target_stoi is not None:
            return self.target_stoi
        
        try:
            from pystoi import stoi
            
            # Загружаем оба файла
            processed, sr_proc = librosa.load(str(processed_file), sr=None, mono=True)
            original, sr_orig = librosa.load(str(original_file), sr=None, mono=True)
            
            # Убеждаемся, что частота дискретизации одинакова
            if sr_proc != sr_orig:
                target_sr = min(sr_proc, sr_orig)
                if sr_proc != target_sr:
                    processed = librosa.resample(processed, orig_sr=sr_proc, target_sr=target_sr)
                if sr_orig != target_sr:
                    original = librosa.resample(original, orig_sr=sr_orig, target_sr=target_sr)
                sr = target_sr
            else:
                sr = sr_proc
            
            # Обрезаем до минимальной длины
            min_len = min(len(processed), len(original))
            processed = processed[:min_len]
            original = original[:min_len]
            
            # Вычисляем STOI
            stoi_score = stoi(original, processed, sr, extended=False)
            return float(stoi_score)
        except Exception as e:
            print(f"Ошибка вычисления STOI для {processed_file}: {e}")
            return 0.5  # Значение по умолчанию
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        processed_file, original_file = self.valid_files[idx]
        filename = processed_file.name
        
        # Загружаем аудио
        waveform = self._load_audio(processed_file)
        
        # Парсим параметры из имени файла
        params = self._parse_filename(filename)
        
        # Вычисляем STOI
        stoi_score = self._calculate_stoi(processed_file, original_file)
        
        # Создаем вектор признаков из параметров
        feature_vector = torch.tensor([
            params['snr'] if params['snr'] is not None else 0.0,
            params['rt60'] if params['rt60'] is not None else 0.0,
            params['wet'] if params['wet'] is not None else 0.0
        ], dtype=torch.float32)
        
        return {
            'waveform': waveform,
            'features': feature_vector,
            'stoi': torch.tensor(stoi_score, dtype=torch.float32),
            'filename': filename
        }

