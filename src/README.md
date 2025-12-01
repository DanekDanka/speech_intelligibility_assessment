# STOI Prediction Model

Модель для предсказания STOI (Short-Time Objective Intelligibility) на основе аудио сигналов с использованием предобученного wav2vec2.

## Структура проекта

- `dataset.py` - Dataset класс для загрузки аудио файлов и извлечения признаков
- `model.py` - Архитектура модели на основе wav2vec2
- `train.py` - Скрипт для тренировки модели
- `test.py` - Скрипт для тестирования модели

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Использование

### Тренировка модели

```bash
python train.py \
    --audio_dir /path/to/processed/audio \
    --original_dir /path/to/original/audio \
    --output_dir ./checkpoints \
    --batch_size 8 \
    --epochs 10 \
    --lr 1e-4
```

### Тестирование модели

```bash
python test.py \
    --checkpoint ./checkpoints/best_model.pt \
    --audio_dir /path/to/processed/audio \
    --original_dir /path/to/original/audio \
    --output_dir ./test_results
```

## Параметры модели

- **wav2vec_model**: Предобученная модель wav2vec2 (по умолчанию: `facebook/wav2vec2-base`)
- **hidden_dim**: Размерность скрытых слоев (по умолчанию: 256)
- **num_layers**: Количество полносвязных слоев (по умолчанию: 3)
- **dropout**: Dropout rate (по умолчанию: 0.1)

## Формат данных

Модель ожидает аудио файлы с форматом имени:
- `snr=12_34__name=original_filename.wav` (только шум)
- `rt60=0_87__wet=0_5__name=original_filename.wav` (только реверберация)
- `snr=12_34__rt60=0_87__wet=0_5__name=original_filename.wav` (шум + реверберация)

## Метрики

Модель оценивается по следующим метрикам:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score
- Correlation

