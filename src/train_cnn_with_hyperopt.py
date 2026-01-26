"""
Скрипт для обучения CNN модели с подбором гиперпараметров.
Использует grid search для поиска лучших гиперпараметров,
затем обучает финальную модель на лучших параметрах.
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
from itertools import product

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


def train_single_config(model_kwargs, train_loader, val_loader, device, epochs, lr, config_name):
    """Обучает модель с заданной конфигурацией гиперпараметров"""
    if isinstance(device, str):
        device = torch.device(device)
    
    model = CNNSTOIPredictor(**model_kwargs).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, min_lr=1e-6
    )
    
    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None
    
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    best_val_corr = 0.0
    patience = 3
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        train_loss, train_mae, scaler = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_amp=use_amp, scaler=scaler
        )
        
        val_loss, val_mae, val_corr = validate(
            model, val_loader, criterion, device, use_amp=use_amp
        )
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
            best_val_corr = val_corr
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return {
        'config_name': config_name,
        'best_val_loss': best_val_loss,
        'best_val_mae': best_val_mae,
        'best_val_corr': best_val_corr,
        'model_kwargs': model_kwargs,
        'lr': lr
    }


def generate_conv_configs():
    """Генерирует конфигурации свёрточных слоёв с оптимизацией для R2 > 90%"""
    configs = []
    
    # 4 слоя (ЛУЧШИЕ текущие) - расширяем варианты для улучшения до R2 > 90%
    # Текущий лучший: [128, 256, 512, 1024], kernel=[11, 9, 7, 5], stride=3, lr=5e-5
    
    # Вариация лучшего с большей емкостью
    configs.append({
        'num_filters': [128, 256, 512, 1024],
        'kernel_sizes': [11, 9, 7, 5],  # Лучшая конфигурация - базовая
        'num_conv_layers': 4
    })
    configs.append({
        'num_filters': [160, 320, 640, 1280],  # Больше фильтров
        'kernel_sizes': [11, 9, 7, 5],
        'num_conv_layers': 4
    })
    configs.append({
        'num_filters': [144, 288, 576, 1152],  # Промежуточный вариант
        'kernel_sizes': [11, 9, 7, 5],
        'num_conv_layers': 4
    })
    configs.append({
        'num_filters': [128, 256, 512, 1024],
        'kernel_sizes': [13, 11, 9, 7],  # Еще большие ядра
        'num_conv_layers': 4
    })
    configs.append({
        'num_filters': [192, 384, 768, 1536],  # Максимальная емкость для 4 слоев
        'kernel_sizes': [11, 9, 7, 5],
        'num_conv_layers': 4
    })
    
    # 5 слоёв - значительная емкость
    configs.append({
        'num_filters': [96, 192, 384, 768, 1536],
        'kernel_sizes': [11, 9, 7, 5, 3],
        'num_conv_layers': 5
    })
    configs.append({
        'num_filters': [128, 256, 512, 1024, 2048],
        'kernel_sizes': [11, 9, 7, 5, 3],
        'num_conv_layers': 5
    })
    configs.append({
        'num_filters': [112, 224, 448, 896, 1792],
        'kernel_sizes': [11, 9, 7, 5, 3],
        'num_conv_layers': 5
    })
    configs.append({
        'num_filters': [160, 320, 640, 1280, 2560],
        'kernel_sizes': [11, 9, 7, 5, 3],
        'num_conv_layers': 5
    })
    configs.append({
        'num_filters': [128, 256, 512, 1024, 2048],
        'kernel_sizes': [13, 11, 9, 7, 5],  # Большие ядра для глубокой сети
        'num_conv_layers': 5
    })
    
    # 6 слоёв - максимальная глубина для очень хорошего контекста
    configs.append({
        'num_filters': [64, 128, 256, 512, 1024, 2048],
        'kernel_sizes': [11, 9, 7, 5, 3, 3],
        'num_conv_layers': 6
    })
    configs.append({
        'num_filters': [96, 192, 384, 768, 1536, 3072],
        'kernel_sizes': [11, 9, 7, 5, 3, 3],
        'num_conv_layers': 6
    })
    configs.append({
        'num_filters': [128, 256, 512, 1024, 2048, 4096],
        'kernel_sizes': [11, 9, 7, 5, 3, 3],
        'num_conv_layers': 6
    })
    
    return configs


def hyperparameter_search(train_loader, val_loader, device, epochs, output_dir):
    """Выполняет оптимизированный поиск гиперпараметров для достижения R2 > 90%"""
    
    # Генерируем конфигурации свёрточных слоёв с оптимизацией
    conv_configs = generate_conv_configs()
    
    # Определяем оптимизированное пространство поиска гиперпараметров
    # На основе лучшей конфигурации: stride=3, dropout=0.05, fc_hidden_dim=384, 
    # num_fc_layers=2, lr=5e-5 (текущая корреляция 0.9284)
    hyperparameter_space = {
        'stride': [2, 3],  # Оба значения работают хорошо
        'dropout': [0.02, 0.04, 0.05, 0.06, 0.08],  # Фокус на низкие значения для лучшей емкости
        'fc_hidden_dim': [256, 320, 384, 512, 640],  # Расширяем вокруг лучшего 384
        'num_fc_layers': [2, 3, 4],  # Концентрируемся на 2-4 слоях
        'lr': [3e-5, 4e-5, 5e-5, 6e-5, 8e-5],  # Фокус вокруг лучшего 5e-5
    }
    
    # Генерируем все комбинации гиперпараметров
    keys = list(hyperparameter_space.keys())
    values = list(hyperparameter_space.values())
    hyperparam_combinations = list(product(*values))
    
    # Комбинируем конфигурации свёрточных слоёв с гиперпараметрами
    all_configs = []
    for conv_config in conv_configs:
        for hyperparam_combo in hyperparam_combinations:
            config_dict = dict(zip(keys, hyperparam_combo))
            config_dict.update(conv_config)
            all_configs.append(config_dict)
    
    # Ограничиваем количество комбинаций для ускорения
    # Увеличиваем до 60 конфигураций для более тщательного поиска (цель: R2 > 90%)
    max_configs = 60
    if len(all_configs) > max_configs:
        print(f"Ограничиваем поиск до {max_configs} конфигураций из {len(all_configs)}")
        # Берем равномерно распределенные конфигурации
        step = len(all_configs) // max_configs
        all_configs = all_configs[::step][:max_configs]
    
    print(f"\n{'='*80}")
    print(f"Оптимизированный поиск гиперпараметров для улучшения R2 > 90%")
    print(f"Конфигураций свёрточных слоёв: {len(conv_configs)}")
    print(f"Комбинаций гиперпараметров: {len(hyperparam_combinations)}")
    print(f"Всего конфигураций для проверки: {len(all_configs)}")
    print(f"Текущий лучший результат: R2 ≈ 0.862 (корр: 0.9284)")
    print(f"Целевой результат: R2 > 0.90 (корр: > 0.9487)")
    print(f"{'='*80}\n")
    
    results = []
    
    for idx, config_dict in enumerate(all_configs, 1):
        # Формируем model_kwargs
        model_kwargs = {
            'input_dim': 1,
            'num_filters': config_dict['num_filters'],
            'kernel_sizes': config_dict['kernel_sizes'],
            'stride': config_dict['stride'],
            'dropout': config_dict['dropout'],
            'use_metadata_features': False,
            'fc_hidden_dim': config_dict['fc_hidden_dim'],
            'num_fc_layers': config_dict['num_fc_layers']
        }
        
        lr = config_dict['lr']
        num_conv_layers = config_dict['num_conv_layers']
        config_name = f"config_{idx}_conv{num_conv_layers}"
        
        print(f"\n[{idx}/{len(all_configs)}] {config_name}")
        print(f"  Conv слоёв: {num_conv_layers}, filters: {model_kwargs['num_filters']}")
        print(f"  kernels: {model_kwargs['kernel_sizes']}, stride: {model_kwargs['stride']}")
        print(f"  dropout: {model_kwargs['dropout']}, fc_dim: {model_kwargs['fc_hidden_dim']}, "
              f"fc_layers: {model_kwargs['num_fc_layers']}, lr: {lr:.0e}")
        
        try:
            result = train_single_config(
                model_kwargs, train_loader, val_loader, device, epochs, lr, config_name
            )
            results.append(result)
            
            r2_approx = result['best_val_corr'] ** 2
            marker = "✓✓" if r2_approx > 0.90 else ("✓" if r2_approx > 0.86 else "✗")
            print(f"  {marker} Val Loss: {result['best_val_loss']:.6f}, "
                  f"Val MAE: {result['best_val_mae']:.6f}, "
                  f"R2: {r2_approx:.4f} (корр: {result['best_val_corr']:.4f})")
        except Exception as e:
            print(f"  ✗ Ошибка: {e}")
            continue
    
    # Сортируем результаты по val_loss
    results.sort(key=lambda x: x['best_val_loss'])
    
    # Сохраняем результаты поиска
    search_results_file = os.path.join(output_dir, 'hyperparameter_search_results.json')
    with open(search_results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_configs_tested': len(results),
            'results': results
        }, f, indent=2)
    
    # Выводим топ-10 лучших конфигураций
    print(f"\n{'='*80}")
    print(f"Поиск гиперпараметров завершен")
    print(f"Протестировано конфигураций: {len(results)}")
    print(f"\nТоп-10 лучших конфигураций:")
    print(f"{'='*80}")
    for i, result in enumerate(results[:10], 1):
        r2_approx = result['best_val_corr'] ** 2
        marker = "✓✓" if r2_approx > 0.90 else "✓"
        print(f"\n{i}. {marker} {result['config_name']} | R2: {r2_approx:.4f} | Corr: {result['best_val_corr']:.4f}")
        print(f"   Loss: {result['best_val_loss']:.6f}, MAE: {result['best_val_mae']:.6f}")
        print(f"   Conv: {len(result['model_kwargs']['num_filters'])} слоев, "
              f"filters: {result['model_kwargs']['num_filters']}")
        print(f"   kernels: {result['model_kwargs']['kernel_sizes']}, stride: {result['model_kwargs']['stride']}")
        print(f"   dropout: {result['model_kwargs']['dropout']}, fc_dim: {result['model_kwargs']['fc_hidden_dim']}, "
              f"fc_layers: {result['model_kwargs']['num_fc_layers']}, lr: {result['lr']:.0e}")
    
    print(f"\n{'='*80}")
    print(f"Результаты сохранены в: {search_results_file}")
    print(f"{'='*80}\n")
    
    return results


def train_final_model(best_config, train_loader, val_loader, device, epochs, output_dir):
    """Обучает финальную модель на лучших гиперпараметрах с улучшенной стабильностью"""
    print(f"\n{'='*80}")
    print(f"Обучение финальной модели на оптимизированных гиперпараметрах")
    print(f"{'='*80}\n")
    
    model_kwargs = best_config['model_kwargs']
    lr = best_config['lr']
    
    print("Оптимизированные гиперпараметры:")
    print(f"  Архитектура: {len(model_kwargs['num_filters'])} conv слоев")
    print(f"  num_filters: {model_kwargs['num_filters']}")
    print(f"  kernel_sizes: {model_kwargs['kernel_sizes']}")
    print(f"  stride: {model_kwargs['stride']}")
    print(f"  dropout: {model_kwargs['dropout']}")
    print(f"  fc_hidden_dim: {model_kwargs['fc_hidden_dim']}")
    print(f"  num_fc_layers: {model_kwargs['num_fc_layers']}")
    print(f"  learning_rate: {lr:.0e}")
    print()
    
    if isinstance(device, str):
        device = torch.device(device)
    
    model = CNNSTOIPredictor(**model_kwargs).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров модели: {num_params:,}\n")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999))
    
    # Улучшенный scheduler для лучшей сходимости
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.6, patience=4, min_lr=1e-7
    )
    
    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda') if use_amp else None
    
    log_dir = os.path.join(output_dir, 'logs_cnn_final')
    writer = SummaryWriter(log_dir=log_dir)
    
    best_val_loss = float('inf')
    best_val_mae = float('inf')
    best_val_corr = 0.0
    best_r2 = 0.0
    patience = 6  # Увеличенное терпение для стабильного обучения
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': [],
        'val_corr': [],
        'val_r2': []
    }
    
    for epoch in range(1, epochs + 1):
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
        
        print(f"Epoch {epoch}/{epochs}")
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
                'model_name': 'cnn',
                'best_hyperparameters': best_config
            }
            
            checkpoint_path = os.path.join(output_dir, 'best_cnn_hyperopt.pt')
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
    best_hyperparams_file = os.path.join(output_dir, 'best_hyperparameters.json')
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
    
    print(f"\nЛучшие гиперпараметры сохранены в: {best_hyperparams_file}")
    print(f"Финальная модель сохранена в: {os.path.join(output_dir, 'best_cnn_hyperopt.pt')}")
    
    return {
        'best_val_loss': best_val_loss,
        'best_val_mae': best_val_mae,
        'best_val_corr': best_val_corr,
        'best_r2': best_r2,
        'model_kwargs': model_kwargs,
        'lr': lr
    }


def main():
    parser = argparse.ArgumentParser(description='Обучение CNN модели с подбором гиперпараметров')
    parser.add_argument('--audio_dir', type=str, 
                       default='/home/danya/datasets/CMU-MOSEI/Audio/',
                       help='Базовая директория с обработанными аудио файлами')
    parser.add_argument('--original_dir', type=str,
                       default='/home/danya/datasets/CMU-MOSEI/Audio/WAV_16000/',
                       help='Директория с оригинальными аудио файлами')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_cnn_hyperopt_v2',
                       help='Директория для сохранения чекпоинтов')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Размер батча')
    parser.add_argument('--search_epochs', type=int, default=20,
                       help='Количество эпох для каждой конфигурации при поиске')
    parser.add_argument('--final_epochs', type=int, default=100,
                       help='Количество эпох для финального обучения')
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
    parser.add_argument('--skip_search', action='store_true', default=False,
                       help='Пропустить поиск гиперпараметров и использовать сохраненные')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Поиск гиперпараметров
    if not args.skip_search:
        search_results = hyperparameter_search(
            train_loader, val_loader, args.device, args.search_epochs, args.output_dir
        )
        best_config = search_results[0]  # Лучшая конфигурация (первая после сортировки)
    else:
        # Загружаем сохраненные лучшие гиперпараметры
        best_hyperparams_file = os.path.join(args.output_dir, 'best_hyperparameters.json')
        if not os.path.exists(best_hyperparams_file):
            raise FileNotFoundError(f"Файл с гиперпараметрами не найден: {best_hyperparams_file}")
        
        with open(best_hyperparams_file, 'r') as f:
            hyperparams_data = json.load(f)
        
        best_config = {
            'model_kwargs': hyperparams_data['hyperparameters']['model_kwargs'],
            'lr': hyperparams_data['hyperparameters']['lr'],
            'best_val_loss': hyperparams_data.get('best_val_loss', float('inf'))
        }
        print(f"Загружены сохраненные гиперпараметры из {best_hyperparams_file}")
    
    # Обучение финальной модели
    final_result = train_final_model(
        best_config, train_loader, val_loader, args.device, args.final_epochs, args.output_dir
    )
    
    print("\n" + "="*80)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*80)
    print(f"Лучшая модель сохранена в: {os.path.join(args.output_dir, 'best_cnn_hyperopt.pt')}")
    print(f"Лучшие гиперпараметры сохранены в: {os.path.join(args.output_dir, 'best_hyperparameters.json')}")
    print(f"\nФинальные метрики:")
    print(f"  Val Loss: {final_result['best_val_loss']:.6f}")
    print(f"  Val MAE: {final_result['best_val_mae']:.6f}")
    print(f"  Val Corr: {final_result['best_val_corr']:.6f}")
    print(f"  R2 Score: {final_result.get('best_r2', final_result['best_val_corr']**2):.6f}")
    
    # Проверка достижения целевого результата
    r2 = final_result.get('best_r2', final_result['best_val_corr']**2)
    if r2 > 0.90:
        print(f"\n✓✓ УСПЕХ! Достигнут целевой R2 > 0.90!")
    elif r2 > 0.86:
        print(f"\n✓ Хороший результат R2 > 0.86, близко к целевому R2 > 0.90")


if __name__ == '__main__':
    main()

