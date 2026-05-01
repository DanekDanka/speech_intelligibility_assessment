# Модель и данные: конфиг `train_stoi_net.json`

Документ описывает **неинтрузивный** предсказатель STOI в проекте `src_STOI`: сеть `stoi_net_predictor` (STOI‑Net, по [Zezario et al., STOI‑Net, arXiv:2011.04292](https://arxiv.org/abs/2011.04292)), датасеты и гиперпараметры из `src_STOI/configs/train_stoi_net.json`.

## Задача

На вход — моно-волна (waveform) деградированной речи (5 с при 16 kHz в обучении). На выход — **один скаляр** в диапазоне **(0, 1)**, оценка интеллигибельности в терминах STOI. Метка при обучении — **референсный STOI** (чистый сигнал из `original_dir` / VoxCeleb mirror, деградированный — текущий файл), считаемый `TorchaudioBackedStoiTargetComputer` (реализация согласована с `torchaudio.functional.stoi` при наличии, иначе Taal + ресемплинг). Функция потерь — **MSE** между предсказанием и таргетом.

---

## Архитектура `StoiNetPredictor` (`stoi_net_predictor`)

1. **STFT** (окно Хэмминга, `center=True`, one-sided): амплитудный спектр; при `use_log_mag: true` — `log1p(|S|)`.
2. **12 слоёв Conv2D** (3×3, `BatchNorm2d`, ReLU) с периодическим `stride` по **оси частоты** (2 на слоях 2, 6, 10) — сжатие по bins, **время без stride**.
3. **Global average pool** по оси частоты до 1 бина.
4. **BiLSTM** (последовательность по временным кадрам).
5. **Поэлементная multiplicative attention** (sigmoid от линейного проекта скрытого состояния).
6. **Полносвязная голова** на каждом кадре → **среднее по времени** → **sigmoid** (как и в других предсказателях в репозитории — один loss на весь чанк, не frame-level из статьи).

Каналы свёрток по умолчанию (12 чисел, три раза 16, 32, 64, 128) заданы в `src_STOI/model/stoi_net_predictor.py`; менять `conv_channels` в конфиге не требуется, если не экспериментируете с архитектурой.

---

## Размерности тензоров (как в коде, для типичного чанка обучения)

Параметры: `n_fft=512` → `F₀ = n_fft/2+1 = 257` бинов по частоте; `hop_length=256` → **313** кадров STFT на **T = 5 c × 16000 = 80 000** сэмплов (см. `_spectrogram` и `forward` в `stoi_net_predictor.py`). `B` — размер батча.

| Этап | Форма | Пояснение |
|------|--------|------------|
| Вход | `(B, T)` | `T` задаёт `chunk_duration_sec` × `sample_rate` |
| Спектр (магнитуда / log) | `(B, 257, 313)` | `257` bins, `313` frames |
| После `unsqueeze(1)` | `(B, 1, 257, 313)` | 1 «канал» |
| После стека CNN (12 conv) | `(B, 128, 33, 313)` | 128 каналов; 257 → 33 по частоте (три `stride=2` по freq) |
| `adaptive_avg_pool2d(…, (1, None))` | `(B, 128, 1, 313)` | усреднение по частоте в один ряд |
| Подготовка к RNN (squeeze+transpose) | `(B, 313, 128)` | 313 **временных шагов** для LSTM, признак 128 |
| Выход BiLSTM | `(B, 313, 256)` | `2 × lstm_hidden` |
| Линей + attention, затем `frame_fc` | `(B, 313, 1)` | скаляр на кадр |
| Усреднение по времени + sigmoid | `(B, 1)` | итоговая оценка STOI |

**Число обучаемых параметров** (конфиг по умолчанию, без `conv_channels`): **692 481** (можно увидеть при старте `src_STOI/train.py` и проверить через `sum(p.numel() for p in model.parameters())`).

Если изменить `chunk_duration_sec` или `sample_rate`, длина `T` и число STFT-кадров (313 для 5 s) изменятся; **ось «313»** в таблице заменить на `n_frames` из фактического STFT.

---

## Параметры модели из `train_stoi_net.json` и что ими регулируют

| Параметр | Значение | Роль / влияние |
|----------|----------|----------------|
| `sample_rate` | 16000 | Согласование с волнами в датасете и STFT |
| `n_fft` | 512 | Число бинов по частоте, частотное разрешение |
| `hop_length` | 256 | Шаг STFT, плотность **временных** кадров |
| `win_length` | 512 | Длина окна (обычно = `n_fft` здесь) |
| `lstm_hidden` | 128 | Ёмкость BiLSTM; рост — больше VRAM, сильнее последовательная модель |
| `lstm_layers` | 1 | Глубина LSTM; при >1 ещё используется `dropout` между слоями |
| `fc_dim` | 128 | Размер скрытого слоя `frame_fc` |
| `dropout` | 0.1 | Регуляризация LSTM/FC; при 1 LSTM применяется к выходу FC-слоя |
| `use_log_mag` | true | `log1p` от магнитуды: устойчивее к динамическому диапазону |

`conv_channels` в JSON не заданы — в коде используется набор **по умолчанию** (12 слоёв, см. исходник). Явно задать список из 12 целых можно, если переносите другую вариацию сети из той же статьи.

---

## Обучение (из того же JSON)

- Оптимизатор **Adam**, `lr: 0.0001`, `weight_decay: 0`.
- **Потеря:** `torchaudio_stoi` → MSE(предсказание, target).
- 10 **эпох**, `batch_size: 12`, `num_workers: 4`, **AMP**включён, `grad_clip: 1.0`.
- `output_dir`: `./checkpoints_src_stoi_net`.

**Целевой STOI (блок `stoi_target`):** `extended: false` (стандартный STOI, не eSTOI-расширение), `resample_mode: "torchaudio"`, `compute_device: "cuda"`, `apply_silence_removal: false` (весь чанк участвует, без VAD-отсекания тишины в Taal).

---

## Датасет: CMU-MOSEI + VoxCeleb (augmented)

### Общая схема

- Для **каждой** деградированной записи в подпапках `processed` ищется **пара** с **чистым** референсом с тем же логичным именем/зеркалом (см. `PairWavStoiDataset` в `src_STOI/data/pair_wav_dataset.py`).
- Чанки: длина **5.0 s**, `sample_rate` **16 000**; на файл кладётся `ceil(длина_в_сэмплах / 80000)` чанков (если `single_chunk_per_file: false`).
- `index_fraction: 1.0` — в индекс попадают все чанки; `train_data_fraction: 1.0` — весь train после сплитов; `shuffle_chunks: true` — порядок чанков в датасете перемешан.
- `train_on_all_data: true`, `val_holdout_fraction: 0.01` — почти все **чанки** CMU и Vox train в обучении, **1%** от каждой части (отдельные permutations) — **валидация**; `test` при `voxceleb.enabled: true` **только** VoxCeleb **test** (augmented), см. `train.py`.

### CMU-MOSEI (`data` в JSON)

- `audio_base_dir`: `…/CMU-MOSEI/Audio`
- `subdirs`: `noise`, `reverb`, `noise_reverb`, `extreme_stoi` — **отдельные наборы** с разной деградацией, все подмешиваются в один индекс.
- `original_dir`: `…/CMU-MOSEI/Audio/WAV_16000` — чистые 16 kHz.
- **Пороги аугментации** (скрипты `scripts/`, согласовано с Vox, кроме `extreme_stoi`):
  - **noise:** белый шум, SNR **−10 … 20 dB**, **30** бинов, равномерное по бинам.
  - **reverb / noise_reverb:** RT60 **0.1 … 2.0 s** (**30** бинов), **wet_level 0.1 … 0.8**; для **noise**/**noise_reverb** — те же SNR, что выше.
  - **extreme_stoi** (см. `create_extreme_stoi_dataset.py`): **~2000** целевых клипов — **высокий** STOI (в т.ч. SNR 20…35 dB) и **низкий** STOI (сильная реверберация, RT60 **3.5 … 5.0 s**, wet **0.9 … 0.98**).

**Снимок по числу аудиофайлов** (подсчёт `.wav` / `.m4a` на диске по путям из вашего `train_stoi_net.json`, 2026):

| Расположение | ~Число файлов |
|--------------|---------------|
| `WAV_16000` (чистые) | 3 842 |
| `noise/` | 3 842 |
| `reverb/` | 3 842 |
| `noise_reverb/` | 3 842 |
| `extreme_stoi/` | 3 876 |

*Точное число **чанков** CMU* больше, чем число файлов (длинные записи дают несколько 5 s окон) — смотрите логи `train.py` («всего чанков=…»).

### VoxCeleb (`voxceleb` в JSON, `enabled: true`)

- `mirror_root`: **оригинальные** клипы; `augmented_root` — **зашумлённые/реверб. копии** (та же схема, что и для MOSEI, см. `add_noise_reverb_voxceleb.py`):
  - SNR **−10 … 20 dB** (30 бинов);
  - RT60 **0.1 … 2.0 s** (30 бинов);
  - wet **0.1 … 0.8**.
- Train: `train/noise`, `train/reverb`, `train/noise_reverb` — **отдельно** каталоги с одним типом деградации (имена/парсинг согласованы с вьюверами в репо).
- Test: `test/noise`, `test/reverb`, `test/noise_reverb`.

**Снимок по подсчёту на тех же путях:**

| Каталог | ~Число файлов |
|---------|----------------|
| `train/noise` | 297 233 |
| `train/reverb` | 297 272 |
| `train/noise_reverb` | 297 284 |
| `test/noise` | 47 206 |
| `test/reverb` | 47 207 |
| `test/noise_reverb` | 47 207 |

`voxceleb.max_items` / `max_items_test` в конфиге **`null`** — **без** искусственного среза; Vox **доминирует** по объёму чанков относительно CMU.

**Кэши:** `cache_stoi_path: ./stoi_label_cache.pkl`, `stoi_rich_cache_path: ./stoi_train_partly_rich_cache.pkl` (при полном прогоне сначала может долго заполняться; повторы быстрее). Импорт легаси: `stoi_import_legacy_cache_paths: ["./stoi_label_cache.pkl"]`.

---

## Сводка

Конфиг `src_STOI/configs/train_stoi_net.json` задаёт **STOI‑Net** с фиксированной цепочкой **STFT → 12×Conv → BiLSTM+attention → FC → усреднение** и **~6.9×10⁵** параметров, обучение **MSE** к интрушивно посчитанному STOI. **Данные** — **четыре** типа деградации **CMU-MOSEI** + **три** больших **VoxCeleb** train и отдельный Vox test; цифры **файлов** выше отражают **текущее** дерево в `audio_base_dir` / `augmented_root` на машине, где ведётся обучение; **чанки** смотрите в выводе `python src_STOI/train.py --config src_STOI/configs/train_stoi_net.json` после строки `Готово (основной датасет)…` и `Vox train чанков=…`.
