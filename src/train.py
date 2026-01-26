"""
Скрипт для тренировки модели предсказания STOI
"""
import os
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

from dataset import STOIDataset
from model import STOIPredictor

# Подавляем предупреждения о градиентах для замороженных моделей
warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad=True.*")


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Одна эпоха тренировки"""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        waveform = batch['waveform'].to(device)
        stoi_true = batch['stoi'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        stoi_pred = model(waveform)
        
        # Убеждаемся, что размеры совпадают
        stoi_pred_flat = stoi_pred.view(-1)  # (batch_size,)
        stoi_true_flat = stoi_true.view(-1)  # (batch_size,)
        
        # Вычисляем loss
        loss = criterion(stoi_pred_flat, stoi_true_flat)
        
        # Backward pass
        loss.backward()
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
    
    return avg_loss, avg_mae


def validate(model, dataloader, criterion, device):
    """Валидация модели"""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    all_preds = []
    all_targets = []
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='[Val]')
        for batch in pbar:
            waveform = batch['waveform'].to(device)
            stoi_true = batch['stoi'].to(device)
            
            # Forward pass
            stoi_pred = model(waveform)
            
            # Убеждаемся, что размеры совпадают
            stoi_pred_flat = stoi_pred.view(-1)  # (batch_size,)
            stoi_true_flat = stoi_true.view(-1)  # (batch_size,)
            
            # Вычисляем loss
            loss = criterion(stoi_pred_flat, stoi_true_flat)
            
            # Метрики
            mae = torch.abs(stoi_pred_flat - stoi_true_flat).mean().item()
            total_loss += loss.item()
            total_mae += mae
            num_batches += 1
            
            # Сохраняем предсказания для анализа
            # Используем numpy() напрямую, чтобы избежать проблем с 0-d массивами
            preds_np = stoi_pred_flat.cpu().numpy()
            targets_np = stoi_true_flat.cpu().numpy()
            
            # Преобразуем в список, если это скаляр
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


def main():
    parser = argparse.ArgumentParser(description='Тренировка модели предсказания STOI')
    parser.add_argument('--audio_dir', type=str, 
                       default='/home/danya/datasets/CMU-MOSEI/Audio/',
                       help='Базовая директория с обработанными аудио файлами (будет искать в поддиректориях noise, reverb, noise_reverb)')
    parser.add_argument('--original_dir', type=str,
                       default='/home/danya/datasets/CMU-MOSEI/Audio/WAV_16000/',
                       help='Директория с оригинальными аудио файлами')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Директория для сохранения чекпоинтов')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Размер батча')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Количество эпох')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--wav2vec_model', type=str, default='facebook/wav2vec2-base',
                       help='Название модели wav2vec2 (можно использовать wav2vec2-large для лучших результатов, но требуется больше памяти)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Размерность скрытых слоев')
    parser.add_argument('--num_layers', type=int, default=5,
                       help='Количество полносвязных слоев')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--use_residual', action='store_true', default=True,
                       help='Использовать residual connections')
    parser.add_argument('--freeze_wav2vec', action='store_true', default=True,
                       help='Замораживать веса wav2vec (False = fine-tuning, требует больше памяти)')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Доля данных для тренировки')
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Частота дискретизации')
    parser.add_argument('--max_length', type=float, default=10.0,
                       help='Максимальная длина аудио в секундах')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Максимальное количество образцов для тренировки (None = использовать все)')
    parser.add_argument('--single_chunk_per_audio', action='store_true', default=False,
                       help='Брать только один чанк на аудио (для быстрого тестового обучения)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Устройство для тренировки')
    
    args = parser.parse_args()
    
    # Создаем директорию для чекпоинтов
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Создаем dataset
    print("Загрузка датасета...")
    print(f"Базовая директория: {args.audio_dir}")
    print("Ищем файлы в поддиректориях: noise, reverb, noise_reverb, extreme_stoi")
    dataset = STOIDataset(
        audio_dir=args.audio_dir,
        original_dir=args.original_dir,
        sample_rate=args.sample_rate,
        max_length_seconds=args.max_length,
        use_wav2vec=True,
        subdirs=['noise', 'reverb', 'noise_reverb', 'extreme_stoi'],  # Ищем файлы в этих поддиректориях
        single_chunk_per_audio=args.single_chunk_per_audio
    )
    
    # Проверяем, что датасет не пустой
    if len(dataset) == 0:
        raise ValueError(
            f"Датасет пуст! Не найдено валидных файлов.\n"
            f"Проверьте:\n"
            f"  - Путь к обработанным файлам: {args.audio_dir}\n"
            f"  - Путь к оригинальным файлам: {args.original_dir}\n"
            f"  - Формат имен файлов соответствует ожидаемому"
        )
    
    # Ограничиваем размер датасета, если указано
    if args.max_samples is not None and args.max_samples > 0:
        if args.max_samples < len(dataset):
            print(f"Ограничиваем датасет до {args.max_samples} образцов (было {len(dataset)})")
            # Создаем подмножество датасета
            indices = torch.randperm(len(dataset))[:args.max_samples].tolist()
            dataset = torch.utils.data.Subset(dataset, indices)
        else:
            print(f"max_samples ({args.max_samples}) больше размера датасета ({len(dataset)}), используем все образцы")
    
    # Разделяем на train и val
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    if train_size == 0 or val_size == 0:
        raise ValueError(
            f"Недостаточно данных для разделения на train/val. "
            f"Всего файлов: {len(dataset)}, train_split: {args.train_split}"
        )
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Создаем DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True if args.device == 'cuda' else False,
        persistent_workers=True if args.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True if args.device == 'cuda' else False,
        persistent_workers=True if args.device == 'cuda' else False
    )
    
    # Создаем модель
    print("Создание модели...")
    model = STOIPredictor(
        wav2vec_model_name=args.wav2vec_model,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_audio_features=True,
        use_metadata_features=False,
        use_residual=args.use_residual,
        freeze_wav2vec=args.freeze_wav2vec
    ).to(args.device)
    
    print(f"Модель создана. Параметров: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss и optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # Тренировка
    best_val_loss = float('inf')
    
    print("\nНачало тренировки...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch
        )
        
        # Validate
        val_loss, val_mae, val_corr = validate(
            model, val_loader, criterion, args.device
        )
        
        # Scheduler step
        scheduler.step(val_loss)
        
        # Логирование
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('MAE/Train', train_mae, epoch)
        writer.add_scalar('MAE/Val', val_mae, epoch)
        writer.add_scalar('Correlation/Val', val_corr, epoch)
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Train - Loss: {train_loss:.4f}, MAE: {train_mae:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, Corr: {val_corr:.4f}")
        
        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_corr': val_corr,
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"✓ Сохранена лучшая модель (val_loss: {val_loss:.4f})")
        
        # Сохраняем последний чекпоинт
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(checkpoint, os.path.join(args.output_dir, 'last_model.pt'))
    
    writer.close()
    print("\nТренировка завершена!")
    print(f"Лучшая модель сохранена в: {os.path.join(args.output_dir, 'best_model.pt')}")


if __name__ == '__main__':
    main()

