"""
Скрипт для обучения финальной CNN модели с оптимизированными гиперпараметрами.
Использует лучшие параметры, найденные в результате подбора гиперпараметров.
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
from model import CNNSTOIPredictor

warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad=True.*")


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, use_amp=True, scaler=None):
    """Одна эпоха тренировки с поддержкой mixed precision"""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    if isinstance(device, str):
        device = torch.device(device)
    
    if scaler is None and use_amp:
        scaler = GradScaler('cuda')
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]', leave=False)
    for batch in pbar:
        waveform = batch['waveform'].to(device, non_blocking=True)
        stoi_true = batch['stoi'].to(device, non_blocking=True)
        
        waveform = waveform.contiguous()
        
        optimizer.zero_grad()
        
        if use_amp and device.type == 'cuda':
            with autocast(device_type='cuda'):
                stoi_pred = model(waveform)
                stoi_pred_flat = stoi_pred.view(-1)
                stoi_true_flat = stoi_true.view(-1)
                loss = criterion(stoi_pred_flat, stoi_true_flat)
            
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
        
        mae = torch.abs(stoi_pred_flat - stoi_true_flat).mean().item()
        total_loss += loss.item()
        total_mae += mae
        num_batches += 1
        
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
    
    if isinstance(device, str):
        device = torch.device(device)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='[Val]', leave=False)
        for batch in pbar:
            waveform = batch['waveform'].to(device, non_blocking=True)
            stoi_true = batch['stoi'].to(device, non_blocking=True)
            
            waveform = waveform.contiguous()
            
            if use_amp and device.type == 'cuda':
                with autocast(device_type='cuda'):
                    stoi_pred = model(waveform)
            else:
                stoi_pred = model(waveform)
            
            stoi_pred_flat = stoi_pred.view(-1)
            stoi_true_flat = stoi_true.view(-1)
            
            loss = criterion(stoi_pred_flat, stoi_true_flat)
            
            mae = torch.abs(stoi_pred_flat - stoi_true_flat).mean().item()
            total_loss += loss.item()
            total_mae += mae
            num_batches += 1
            
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
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    
    correlation = np.corrcoef(all_preds, all_targets)[0, 1]
    
    return avg_loss, avg_mae, correlation


def main():
    parser = argparse.ArgumentParser(description='Обучение финальной CNN модели с оптимизированными гиперпараметрами')
    parser.add_argument('--audio_dir', type=str, 
                       default='/home/danya/datasets/CMU-MOSEI/Audio/',
                       help='Базовая директория с обработанными аудио файлами')
    parser.add_argument('--original_dir', type=str,
                       default='/home/danya/datasets/CMU-MOSEI/Audio/WAV_16000/',
                       help='Директория с оригинальными аудио файлами')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_cnn_final',
                       help='Директория для сохранения чекпоинтов')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Размер батча')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Количество эпох для обучения')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Доля данных для тренировки')
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Частота дискретизации')
    parser.add_argument('--max_length', type=float, default=10.0,
                       help='Максимальная длина аудио в секундах')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Максимальное количество образцов для тренировки')
    parser.add_argument('--single_chunk_per_audio', action='store_true', default=True,
                       help='Брать только один чанк на аудио')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Устройство для тренировки')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Оптимизированные гиперпараметры (лучшая конфигурация из подбора)
    model_kwargs = {
        'input_dim': 1,
        'num_filters': [96, 192, 384, 768, 1536],  # 5 conv слоев
        'kernel_sizes': [11, 9, 7, 5, 3],
        'stride': 3,
        'dropout': 0.06,
        'use_metadata_features': False,
        'fc_hidden_dim': 320,
        'num_fc_layers': 3
    }
    
    lr = 8e-5
    
    print(f"\n{'='*80}")
    print(f"Обучение финальной CNN модели с оптимизированными гиперпараметрами")
    print(f"{'='*80}\n")
    
    print("Гиперпараметры модели:")
    print(f"  Архитектура: {len(model_kwargs['num_filters'])} conv слоев")
    print(f"  num_filters: {model_kwargs['num_filters']}")
    print(f"  kernel_sizes: {model_kwargs['kernel_sizes']}")
    print(f"  stride: {model_kwargs['stride']}")
    print(f"  dropout: {model_kwargs['dropout']}")
    print(f"  fc_hidden_dim: {model_kwargs['fc_hidden_dim']}")
    print(f"  num_fc_layers: {model_kwargs['num_fc_layers']}")
    print(f"  learning_rate: {lr:.0e}")
    print()
    
    # Создаем dataset
    print("Загрузка датасета...")
    cache_file = Path(args.output_dir) / '.stoi_cache.pkl'
    dataset = STOIDataset(
        audio_dir=args.audio_dir,
        original_dir=args.original_dir,
        sample_rate=args.sample_rate,
        max_length_seconds=args.max_length,
        use_wav2vec=False,
        subdirs=['noise', 'reverb', 'noise_reverb', 'extreme_stoi'],
        cache_stoi=True,
        cache_file=cache_file,
        single_chunk_per_audio=args.single_chunk_per_audio
    )
    
    if len(dataset) == 0:
        raise ValueError(f"Датасет пуст! Проверьте пути: {args.audio_dir}, {args.original_dir}")
    
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
    
    # Создаем DataLoader
    num_workers = min(8, os.cpu_count() or 1)
    
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
    
    # Создаем модель
    if isinstance(args.device, str):
        device = torch.device(args.device)
    else:
        device = args.device
    
    model = CNNSTOIPredictor(**model_kwargs).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nПараметров модели: {num_params:,}\n")
    
    # Оптимизатор и loss
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.6, patience=4, min_lr=1e-7
    )
    
    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None
    
    # TensorBoard логирование
    log_dir = os.path.join(args.output_dir, 'logs_cnn_final')
    writer = SummaryWriter(log_dir=log_dir)
    
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    best_val_corr = 0.0
    best_r2 = 0.0
    patience = 6
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': [],
        'val_corr': [],
        'val_r2': []
    }
    
    print(f"Начинаем обучение на {args.epochs} эпох...\n")
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae, scaler = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_amp=use_amp, scaler=scaler
        )
        
        val_loss, val_mae, val_corr = validate(
            model, val_loader, criterion, device, use_amp=use_amp
        )
        
        val_r2 = val_corr ** 2
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_corr'].append(val_corr)
        history['val_r2'].append(val_r2)
        
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('MAE/Train', train_mae, epoch)
        writer.add_scalar('MAE/Val', val_mae, epoch)
        writer.add_scalar('Correlation/Val', val_corr, epoch)
        writer.add_scalar('R2/Val', val_r2, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"Train - Loss: {train_loss:.6f}, MAE: {train_mae:.6f}")
        print(f"Val   - Loss: {val_loss:.6f}, MAE: {val_mae:.6f}, Corr: {val_corr:.6f}, R2: {val_r2:.6f}")
        print(f"LR: {current_lr:.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
            best_val_corr = val_corr
            best_r2 = val_r2
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_corr': val_corr,
                'val_r2': val_r2,
                'model_kwargs': model_kwargs,
                'history': history,
                'model_name': 'cnn_final',
                'hyperparameters': {
                    'model_kwargs': model_kwargs,
                    'lr': lr
                }
            }
            
            checkpoint_path = os.path.join(args.output_dir, 'best_cnn_final.pt')
            torch.save(checkpoint, checkpoint_path)
            marker = "✓✓" if best_r2 > 0.90 else "✓"
            print(f"{marker} Сохранена лучшая модель (val_loss: {val_loss:.6f}, R2: {best_r2:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping: нет улучшения в течение {patience} эпох")
                break
        print()
    
    writer.close()
    
    # Сохраняем лучшие гиперпараметры в отдельный файл
    best_hyperparams_file = os.path.join(args.output_dir, 'best_hyperparameters.json')
    with open(best_hyperparams_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'best_val_loss': best_val_loss,
            'best_val_mae': best_val_mae,
            'best_val_corr': best_val_corr,
            'best_r2': best_r2,
            'hyperparameters': {
                'model_kwargs': model_kwargs,
                'lr': lr
            }
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"{'='*80}\n")
    print(f"Лучшая модель сохранена в: {os.path.join(args.output_dir, 'best_cnn_final.pt')}")
    print(f"Лучшие гиперпараметры сохранены в: {best_hyperparams_file}")
    print(f"\nФинальные метрики:")
    print(f"  Val Loss: {best_val_loss:.6f}")
    print(f"  Val MAE: {best_val_mae:.6f}")
    print(f"  Val Corr: {best_val_corr:.6f}")
    print(f"  R2 Score: {best_r2:.6f}")
    
    # Проверка достижения целевого результата
    if best_r2 > 0.90:
        print(f"\n✓✓ УСПЕХ! Достигнут целевой R2 > 0.90!")
    elif best_r2 > 0.86:
        print(f"\n✓ Хороший результат R2 > 0.86, близко к целевому R2 > 0.90")


if __name__ == '__main__':
    main()
