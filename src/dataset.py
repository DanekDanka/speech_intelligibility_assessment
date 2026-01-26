"""
Dataset для загрузки аудио файлов и предсказания STOI
"""
import os
import re
import math
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
import pickle
from tqdm import tqdm


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
                 target_stoi=None,
                 subdirs=None,
                 cache_stoi=True,
                 cache_file=None,
                 single_chunk_per_audio=False):
        """
        Args:
            audio_dir: Базовая директория с обработанными аудио файлами или список директорий
            original_dir: Директория с оригинальными аудио файлами (для вычисления STOI)
            sample_rate: Частота дискретизации
            max_length_seconds: Максимальная длина аудио в секундах
            use_wav2vec: Использовать ли wav2vec для извлечения признаков
            target_stoi: Если указан, используется как целевое значение STOI (для тестирования)
            subdirs: Список поддиректорий для поиска файлов (например, ['noise', 'reverb', 'noise_reverb']).
                     Если None, автоматически ищет в поддиректориях noise, reverb, noise_reverb
            single_chunk_per_audio: Если True, берется только один чанк (первый) на аудио
        """
        self.original_dir = Path(original_dir)
        self.sample_rate = sample_rate
        self.chunk_length_seconds = max_length_seconds
        self.chunk_length = int(max_length_seconds * sample_rate)
        if self.chunk_length <= 0:
            raise ValueError("max_length_seconds должен быть > 0")
        self.use_wav2vec = use_wav2vec
        self.single_chunk_per_audio = single_chunk_per_audio
        
        # Обрабатываем audio_dir - может быть строка или список
        if isinstance(audio_dir, (list, tuple)):
            audio_dirs = [Path(d) for d in audio_dir]
        else:
            base_dir = Path(audio_dir)
            # Если subdirs не указан, используем стандартные поддиректории
            if subdirs is None:
                subdirs = ['noise', 'reverb', 'noise_reverb', 'extreme_stoi']
            
            # Собираем список директорий для поиска
            audio_dirs = []
            for subdir in subdirs:
                dir_path = base_dir / subdir
                if dir_path.exists():
                    audio_dirs.append(dir_path)
                else:
                    print(f"Предупреждение: поддиректория {dir_path} не найдена, пропускаем")
            
            # Если не нашли поддиректории, используем базовую директорию
            if len(audio_dirs) == 0:
                if base_dir.exists():
                    audio_dirs = [base_dir]
                    print(f"Используем базовую директорию: {base_dir}")
                else:
                    raise ValueError(f"Директория с аудио файлами не найдена: {base_dir}")
        
        # Проверяем существование директорий
        for audio_dir_path in audio_dirs:
            if not audio_dir_path.exists():
                raise ValueError(f"Директория с аудио файлами не найдена: {audio_dir_path}")
        
        if not self.original_dir.exists():
            raise ValueError(f"Директория с оригинальными файлами не найдена: {self.original_dir}")
        
        # Находим все аудио файлы во всех указанных директориях
        self.audio_files = []
        for audio_dir_path in audio_dirs:
            files = list(audio_dir_path.rglob("*.wav"))
            self.audio_files.extend(files)
            print(f"Найдено {len(files)} аудио файлов в {audio_dir_path}")
        
        print(f"Всего найдено {len(self.audio_files)} аудио файлов")
        
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
        
        # Формируем индексы чанков
        self.chunk_index = []
        for processed_file, original_file in self.valid_files:
            num_chunks = self._estimate_num_chunks(processed_file)
            if self.single_chunk_per_audio:
                self.chunk_index.append((processed_file, original_file, 0))
            else:
                for chunk_idx in range(num_chunks):
                    self.chunk_index.append((processed_file, original_file, chunk_idx))
        print(f"Всего чанков: {len(self.chunk_index)} (длина чанка {self.chunk_length_seconds:.2f} сек)")

        # Кэширование STOI для ускорения
        self.cache_stoi = cache_stoi
        if cache_file is None:
            cache_file = self.original_dir.parent / '.stoi_cache.pkl'
        self.cache_file = Path(cache_file)
        
        # Загружаем или создаем кэш STOI
        self.stoi_cache = self._load_or_create_stoi_cache()
        
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
    
    def _estimate_num_chunks(self, filepath):
        """Оценивает количество чанков по длительности файла"""
        try:
            info = None
            if hasattr(torchaudio, "info"):
                info = torchaudio.info(str(filepath))
            elif hasattr(torchaudio, "backend") and hasattr(torchaudio.backend, "sox_io_backend"):
                info = torchaudio.backend.sox_io_backend.info(str(filepath))

            if info is not None and info.sample_rate > 0:
                duration_sec = info.num_frames / info.sample_rate
            else:
                waveform, sr = torchaudio.load(str(filepath), normalize=True)
                duration_sec = waveform.shape[1] / sr if sr > 0 else 0.0

            if duration_sec <= 0:
                return 1
            return max(1, math.ceil(duration_sec / self.chunk_length_seconds))
        except Exception as e:
            print(f"Не удалось получить длительность {filepath}: {e}")
            return 1

    def _load_audio_chunk(self, filepath, chunk_idx):
        """Загружает аудио чанк фиксированной длины"""
        try:
            # Используем torchaudio для быстрой загрузки
            waveform, sr = torchaudio.load(str(filepath), normalize=True)
            
            # Конвертируем в моно
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Ресемплируем если нужно
            if sr != self.sample_rate:
                # Создаем ресемплер только если нужно и если его еще нет или частота изменилась
                if (not hasattr(self, '_resampler') or 
                    self._resampler is None or 
                    not hasattr(self._resampler, 'orig_freq') or 
                    self._resampler.orig_freq != sr):
                    self._resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = self._resampler(waveform)
            
            waveform = waveform.squeeze(0)  # (seq_len,)

            # Вырезаем нужный чанк
            start = chunk_idx * self.chunk_length
            end = start + self.chunk_length
            if start >= waveform.shape[0]:
                chunk = torch.zeros(self.chunk_length)
            else:
                chunk = waveform[start:end]
            
            # Дополняем до фиксированной длины
            if chunk.shape[0] < self.chunk_length:
                padding = self.chunk_length - chunk.shape[0]
                chunk = torch.nn.functional.pad(chunk, (0, padding))
            
            return chunk
        except Exception as e:
            print(f"Ошибка загрузки {filepath}: {e}")
            return torch.zeros(self.chunk_length)
    
    def _load_or_create_stoi_cache(self):
        """Загружает существующий кэш STOI или создает новый"""
        cache = {}
        
        # Пробуем загрузить существующий кэш
        if self.cache_file.exists() and self.cache_stoi:
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                print(f"Загружен кэш STOI из {self.cache_file} ({len(cache)} записей)")
            except Exception as e:
                print(f"Не удалось загрузить кэш: {e}, создаем новый")
        
        # Вычисляем STOI для файлов, которых нет в кэше
        if self.cache_stoi and len(self.chunk_index) > 0:
            missing_count = 0
            for processed_file, original_file, chunk_idx in tqdm(self.chunk_index, desc="Вычисление STOI"):
                cache_key = f"{processed_file}|chunk={chunk_idx}"
                if cache_key not in cache:
                    stoi_value = self._calculate_stoi_impl(processed_file, original_file, chunk_idx)
                    cache[cache_key] = stoi_value
                    missing_count += 1
            
            if missing_count > 0:
                print(f"Вычислено {missing_count} новых значений STOI")
                # Сохраняем обновленный кэш
                try:
                    with open(self.cache_file, 'wb') as f:
                        pickle.dump(cache, f)
                    print(f"Кэш STOI сохранен в {self.cache_file}")
                except Exception as e:
                    print(f"Не удалось сохранить кэш: {e}")
        
        return cache
    
    def _calculate_stoi_impl(self, processed_file, original_file, chunk_idx):
        """Внутренняя реализация вычисления STOI"""
        if self.target_stoi is not None:
            return self.target_stoi
        
        try:
            from pystoi import stoi

            processed_chunk = self._load_audio_chunk(processed_file, chunk_idx)
            original_chunk = self._load_audio_chunk(original_file, chunk_idx)

            processed_np = processed_chunk.cpu().numpy()
            original_np = original_chunk.cpu().numpy()

            # Вычисляем STOI на чанке
            stoi_score = stoi(original_np, processed_np, self.sample_rate, extended=False)
            return float(stoi_score)
        except Exception as e:
            print(f"Ошибка вычисления STOI для {processed_file}: {e}")
            return 0.5  # Значение по умолчанию
    
    def _calculate_stoi(self, processed_file, original_file, chunk_idx):
        """Вычисляет STOI между обработанным и оригинальным файлом (с кэшированием)"""
        if self.target_stoi is not None:
            return self.target_stoi
        
        # Используем кэш, если доступен
        if self.cache_stoi:
            cache_key = f"{processed_file}|chunk={chunk_idx}"
            if cache_key in self.stoi_cache:
                return self.stoi_cache[cache_key]
        
        # Если нет в кэше, вычисляем
        return self._calculate_stoi_impl(processed_file, original_file, chunk_idx)
    
    def __len__(self):
        return len(self.chunk_index)
    
    def __getitem__(self, idx):
        processed_file, original_file, chunk_idx = self.chunk_index[idx]
        filename = processed_file.name
        
        # Загружаем аудио
        waveform = self._load_audio_chunk(processed_file, chunk_idx)
        
        # Вычисляем STOI
        stoi_score = self._calculate_stoi(processed_file, original_file, chunk_idx)
        
        return {
            'waveform': waveform,
            'stoi': torch.tensor(stoi_score, dtype=torch.float32),
            'filename': filename,
            'chunk_idx': chunk_idx
        }

