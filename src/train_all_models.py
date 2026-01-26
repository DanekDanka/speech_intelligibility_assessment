"""
Скрипт для тренировки всех моделей (Transformer, LSTM, CNN) и сравнения результатов
"""
import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

from dataset import STOIDataset
from model import TransformerSTOIPredictor, LSTMSTOIPredictor, CNNSTOIPredictor, WhisperSTOIPredictor

warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad=True.*")


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, use_amp=True, scaler=None):
    """Одна эпоха тренировки с поддержкой mixed precision"""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    # Преобразуем device в torch.device если это строка
    if isinstance(device, str):
        device = torch.device(device)
    
    if scaler is None and use_amp:
        scaler = GradScaler('cuda')
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        waveform = batch['waveform'].to(device, non_blocking=True)
        stoi_true = batch['stoi'].to(device, non_blocking=True)
        
        # Убеждаемся, что данные contiguous перед передачей в модель
        waveform = waveform.contiguous()
        
        # Forward pass с mixed precision
        optimizer.zero_grad()
        
        if use_amp and device.type == 'cuda':
            with autocast(device_type='cuda'):
                stoi_pred = model(waveform)
                stoi_pred_flat = stoi_pred.view(-1)
                stoi_true_flat = stoi_true.view(-1)
                loss = criterion(stoi_pred_flat, stoi_true_flat)
            
            # Backward pass с mixed precision
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            stoi_pred = model(waveform)
            stoi_pred_flat = stoi_pred.view(-1)
            stoi_true_flat = stoi_true.view(-1)
            loss = criterion(stoi_pred_flat, stoi_true_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Метрики
        mae = torch.abs(stoi_pred_flat - stoi_true_flat).mean().item()
        total_loss += loss.item()
        total_mae += mae
        num_batches += 1
        
        # Обновляем progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mae': f'{mae:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    
    return avg_loss, avg_mae, scaler


def validate(model, dataloader, criterion, device, use_amp=True):
    """Валидация модели с поддержкой mixed precision"""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0
    
    # Преобразуем device в torch.device если это строка
    if isinstance(device, str):
        device = torch.device(device)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='[Val]')
        for batch in pbar:
            waveform = batch['waveform'].to(device, non_blocking=True)
            stoi_true = batch['stoi'].to(device, non_blocking=True)
            
            # Убеждаемся, что данные contiguous перед передачей в модель
            waveform = waveform.contiguous()
            
            # Forward pass с mixed precision
            if use_amp and device.type == 'cuda':
                with autocast(device_type='cuda'):
                    stoi_pred = model(waveform)
            else:
                stoi_pred = model(waveform)
            
            # Убеждаемся, что размеры правильные
            stoi_pred_flat = stoi_pred.view(-1)
            stoi_true_flat = stoi_true.view(-1)
            
            # Вычисляем loss
            loss = criterion(stoi_pred_flat, stoi_true_flat)
            
            # Метрики
            mae = torch.abs(stoi_pred_flat - stoi_true_flat).mean().item()
            total_loss += loss.item()
            total_mae += mae
            num_batches += 1
            
            # Сохраняем предсказания
            preds_np = stoi_pred_flat.cpu().numpy()
            targets_np = stoi_true_flat.cpu().numpy()
            
            if preds_np.ndim == 0:
                all_preds.append(float(preds_np))
            else:
                all_preds.extend(preds_np.tolist())
            
            if targets_np.ndim == 0:
                all_targets.append(float(targets_np))
            else:
                all_targets.extend(targets_np.tolist())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mae': f'{mae:.4f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    
    # Вычисляем корреляцию
    correlation = np.corrcoef(all_preds, all_targets)[0, 1]
    
    return avg_loss, avg_mae, correlation


def train_model(model_name, model_class, model_kwargs, train_loader, val_loader, 
                device, epochs, lr, output_dir):
    """Обучает одну модель"""
    print(f"\n{'='*60}")
    print(f"Обучение модели: {model_name}")
    print(f"{'='*60}")
    
    # Преобразуем device в torch.device если это строка
    if isinstance(device, str):
        device = torch.device(device)
    
    # Создаем модель
    model = model_class(**model_kwargs).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров модели: {num_params:,}")
    
    # Loss и optimizer
    criterion = nn.MSELoss()
    # Уменьшаем learning rate и увеличиваем weight decay для лучшей стабильности
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))
    
    # Более агрессивный scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )
    
    # Mixed precision training для ускорения
    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None
    if use_amp:
        print("Используется mixed precision training (AMP) для ускорения")
    
    # Early stopping
    patience = 5
    patience_counter = 0
    
    # TensorBoard
    log_dir = os.path.join(output_dir, f'logs_{model_name}')
    writer = SummaryWriter(log_dir=log_dir)
    
    # Тренировка
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    best_val_corr = 0.0
    
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': [],
        'val_corr': []
    }
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_mae, scaler = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_amp=use_amp, scaler=scaler
        )
        
        # Validate
        val_loss, val_mae, val_corr = validate(
            model, val_loader, criterion, device, use_amp=use_amp
        )
        
        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Сохраняем историю
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_corr'].append(val_corr)
        
        # Логирование
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('MAE/Train', train_mae, epoch)
        writer.add_scalar('MAE/Val', val_mae, epoch)
        writer.add_scalar('Correlation/Val', val_corr, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"Train - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, Corr: {val_corr:.4f}")
        print(f"LR: {current_lr:.2e}")
        
        # Сохраняем лучшую модель и проверяем early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
            best_val_corr = val_corr
            patience_counter = 0  # Сбрасываем счетчик при улучшении
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_corr': val_corr,
                'model_kwargs': model_kwargs,
                'history': history,
                'model_name': model_name  # Сохраняем имя модели для удобной загрузки
            }
            
            # Сохраняем лучшую модель с номером эпохи для отслеживания
            checkpoint_path_epoch = os.path.join(output_dir, f'best_{model_name}_epoch{epoch}.pt')
            torch.save(checkpoint, checkpoint_path_epoch)
            
            # Также сохраняем как последнюю лучшую версию (перезаписываем)
            checkpoint_path_best = os.path.join(output_dir, f'best_{model_name}.pt')
            torch.save(checkpoint, checkpoint_path_best)
            
            print(f"✓ Сохранена лучшая модель {model_name} на эпохе {epoch} (val_loss: {val_loss:.4f}, val_mae: {val_mae:.4f})")
            print(f"  - {checkpoint_path_best}")
            print(f"  - {checkpoint_path_epoch}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping: нет улучшения в течение {patience} эпох")
                print(f"Лучшая val_loss: {best_val_loss:.4f} на эпохе {epoch - patience_counter}")
                break
    
    writer.close()
    
    return {
        'model_name': model_name,
        'best_val_loss': best_val_loss,
        'best_val_mae': best_val_mae,
        'best_val_corr': best_val_corr,
        'num_params': num_params,
        'history': history
    }


def main():
    parser = argparse.ArgumentParser(description='Тренировка всех моделей для предсказания STOI')
    parser.add_argument('--audio_dir', type=str, 
                       default='/home/danya/datasets/CMU-MOSEI/Audio/',
                       help='Базовая директория с обработанными аудио файлами (будет искать в поддиректориях noise, reverb, noise_reverb)')
    parser.add_argument('--original_dir', type=str,
                       default='/home/danya/datasets/CMU-MOSEI/Audio/WAV_16000/',
                       help='Директория с оригинальными аудио файлами')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_all_models',
                       help='Директория для сохранения чекпоинтов')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Размер батча (рекомендуется 16-32 для ускорения)')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Количество эпох')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate (рекомендуется 5e-4 для стабильного обучения)')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Доля данных для тренировки')
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Частота дискретизации')
    parser.add_argument('--max_length', type=float, default=10.0,
                       help='Максимальная длина аудио в секундах')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Максимальное количество образцов для тренировки')
    parser.add_argument('--single_chunk_per_audio', action='store_true', default=False,
                       help='Брать только один чанк на аудио (для быстрого тестового обучения)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Устройство для тренировки')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['whisper', 'cnn'],
                       choices=['transformer', 'lstm', 'cnn', 'whisper'],
                       help='Какие модели обучать')
    
    args = parser.parse_args()
    
    # Создаем директорию для чекпоинтов
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Создаем dataset с кэшированием STOI
    print("Загрузка датасета...")
    print(f"Базовая директория: {args.audio_dir}")
    print("Ищем файлы в поддиректориях: noise, reverb, noise_reverb, extreme_stoi")
    cache_file = Path(args.output_dir) / '.stoi_cache.pkl'
    dataset = STOIDataset(
        audio_dir=args.audio_dir,
        original_dir=args.original_dir,
        sample_rate=args.sample_rate,
        max_length_seconds=args.max_length,
        use_wav2vec=False,  # Не используем wav2vec для новых моделей
        subdirs=['noise', 'reverb', 'noise_reverb', 'extreme_stoi'],  # Ищем файлы в этих поддиректориях
        cache_stoi=True,  # Кэшируем STOI для ускорения
        cache_file=cache_file,
        single_chunk_per_audio=args.single_chunk_per_audio
    )
    
    if len(dataset) == 0:
        raise ValueError(f"Датасет пуст! Проверьте пути: {args.audio_dir}, {args.original_dir}")
    
    # Ограничиваем размер датасета, если указано
    if args.max_samples is not None and args.max_samples > 0:
        if args.max_samples < len(dataset):
            print(f"Ограничиваем датасет до {args.max_samples} образцов (было {len(dataset)})")
            indices = torch.randperm(len(dataset))[:args.max_samples].tolist()
            dataset = torch.utils.data.Subset(dataset, indices)
    
    # Разделяем на train и val
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    if train_size == 0 or val_size == 0:
        raise ValueError(f"Недостаточно данных для разделения на train/val")
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Создаем DataLoader с оптимизированными параметрами
    num_workers = min(8, os.cpu_count() or 1)  # Увеличиваем количество воркеров
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if args.device == 'cuda' else False,
        persistent_workers=True if args.device == 'cuda' else False,
        prefetch_factor=2 if args.device == 'cuda' else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if args.device == 'cuda' else False,
        persistent_workers=True if args.device == 'cuda' else False,
        prefetch_factor=2 if args.device == 'cuda' else None
    )
    
    # Определяем модели для обучения
    models_config = {
        'transformer': {
            'class': TransformerSTOIPredictor,
            'kwargs': {
                'input_dim': 1,
                'd_model': 128,  # Уменьшено для экономии памяти
                'nhead': 4,  # Уменьшено для экономии памяти
                'num_layers': 3,  # Уменьшено для экономии памяти
                'dim_feedforward': 512,  # Уменьшено для экономии памяти
                'dropout': 0.1,
                'max_seq_len': int(args.max_length * args.sample_rate),
                'use_metadata_features': False,
                'hidden_dim': 256,  # Уменьшено для экономии памяти
                'num_fc_layers': 2  # Уменьшено для экономии памяти
            }
        },
        'lstm': {
            'class': LSTMSTOIPredictor,
            'kwargs': {
                'input_dim': 1,
                'hidden_dim': 256,
                'num_layers': 3,
                'dropout': 0.2,
                'bidirectional': True,
                'use_metadata_features': False,
                'fc_hidden_dim': 512,
                'num_fc_layers': 3
            }
        },
        'cnn': {
            'class': CNNSTOIPredictor,
            'kwargs': {
                'input_dim': 1,
                'num_filters': [64, 128, 256, 512],
                'kernel_sizes': [7, 5, 3, 3],
                'stride': 2,
                'dropout': 0.2,
                'use_metadata_features': False,
                'fc_hidden_dim': 512,
                'num_fc_layers': 3
            }
        },
        'whisper': {
            'class': WhisperSTOIPredictor,
            'kwargs': {
                'whisper_model_name': 'openai/whisper-base',
                'hidden_dim': 512,
                'num_fc_layers': 5,
                'dropout': 0.2,
                'use_metadata_features': False,
                'freeze_encoder': True,
                'use_residual': True
            }
        }
    }
    
    # Обучаем модели
    results = []
    for model_name in args.models:
        if model_name not in models_config:
            print(f"Пропускаем неизвестную модель: {model_name}")
            continue
        
        config = models_config[model_name]
        result = train_model(
            model_name=model_name,
            model_class=config['class'],
            model_kwargs=config['kwargs'],
            train_loader=train_loader,
            val_loader=val_loader,
            device=args.device,
            epochs=args.epochs,
            lr=args.lr,
            output_dir=args.output_dir
        )
        results.append(result)
    
    # Сохраняем результаты сравнения
    comparison_file = os.path.join(args.output_dir, 'comparison_results.json')
    comparison_data = {
        'timestamp': datetime.now().isoformat(),
        'dataset_size': len(dataset),
        'train_size': len(train_dataset),
        'val_size': len(val_dataset),
        'results': results
    }
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    # Выводим сравнение
    print("\n" + "="*80)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*80)
    print(f"{'Модель':<20} {'Параметры':<15} {'Val Loss':<12} {'Val MAE':<12} {'Val Corr':<12}")
    print("-"*80)
    
    for result in results:
        print(f"{result['model_name']:<20} "
              f"{result['num_params']:>14,} "
              f"{result['best_val_loss']:>11.4f} "
              f"{result['best_val_mae']:>11.4f} "
              f"{result['best_val_corr']:>11.4f}")
    
    print("="*80)
    print(f"\nРезультаты сохранены в: {comparison_file}")
    print("Тренировка всех моделей завершена!")


if __name__ == '__main__':
    main()

