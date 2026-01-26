"""
Скрипт для тестирования модели предсказания STOI
"""
import os
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from dataset import STOIDataset
from model import STOIPredictor

# Подавляем предупреждения о градиентах для замороженных моделей
warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad=True.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*gradient_checkpointing.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*gradient_checkpointing.*")


def test_model(model, dataloader, device):
    """Тестирование модели"""
    model.eval()
    all_preds = []
    all_targets = []
    all_filenames = []
    
    # Очищаем кэш GPU перед тестированием
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Testing')
        for batch_idx, batch in enumerate(pbar):
            waveform = batch['waveform'].to(device)
            stoi_true = batch['stoi'].to(device)
            filenames = batch['filename']
            
            # Forward pass
            stoi_pred = model(waveform)
            
            # Убеждаемся, что размеры правильные
            stoi_pred_flat = stoi_pred.view(-1)  # (batch_size,)
            stoi_true_flat = stoi_true.view(-1)  # (batch_size,)
            
            # Сохраняем результаты (сразу переносим на CPU для экономии памяти)
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
            
            all_filenames.extend(filenames)
            
            # Периодически очищаем кэш GPU (реже, так как больше памяти)
            if device == 'cuda' and (batch_idx + 1) % 50 == 0:
                torch.cuda.empty_cache()
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Вычисляем метрики
    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)
    correlation = np.corrcoef(all_preds, all_targets)[0, 1]
    
    # Вычисляем процент ошибок в пределах порогов
    thresholds = [0.05, 0.10, 0.15, 0.20]
    accuracy_at_threshold = {}
    for threshold in thresholds:
        within_threshold = np.abs(all_preds - all_targets) <= threshold
        accuracy_at_threshold[threshold] = np.mean(within_threshold) * 100
    
    return {
        'predictions': all_preds,
        'targets': all_targets,
        'filenames': all_filenames,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'correlation': correlation,
        'accuracy_at_threshold': accuracy_at_threshold
    }


def plot_results(results, output_dir):
    """Строит графики результатов тестирования"""
    preds = results['predictions']
    targets = results['targets']
    
    # Создаем фигуру с несколькими графиками
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Scatter plot: предсказания vs реальные значения
    axes[0, 0].scatter(targets, preds, alpha=0.5, s=20)
    axes[0, 0].plot([0, 1], [0, 1], 'r--', label='Идеальная линия')
    axes[0, 0].set_xlabel('Реальный STOI', fontsize=12)
    axes[0, 0].set_ylabel('Предсказанный STOI', fontsize=12)
    axes[0, 0].set_title(f'Предсказания vs Реальные значения\nR² = {results["r2"]:.4f}, Corr = {results["correlation"]:.4f}', 
                         fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 2. Гистограмма ошибок
    errors = preds - targets
    axes[0, 1].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Ошибка (Predicted - True)', fontsize=12)
    axes[0, 1].set_ylabel('Количество', fontsize=12)
    axes[0, 1].set_title(f'Распределение ошибок\nMAE = {results["mae"]:.4f}, RMSE = {results["rmse"]:.4f}', 
                         fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=0, color='r', linestyle='--', label='Нет ошибки')
    axes[0, 1].legend()
    
    # 3. Распределение предсказаний и реальных значений
    axes[1, 0].hist(targets, bins=30, alpha=0.6, label='Реальные', edgecolor='black')
    axes[1, 0].hist(preds, bins=30, alpha=0.6, label='Предсказанные', edgecolor='black')
    axes[1, 0].set_xlabel('STOI', fontsize=12)
    axes[1, 0].set_ylabel('Количество', fontsize=12)
    axes[1, 0].set_title('Распределение STOI', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 4. Ошибка в зависимости от STOI
    axes[1, 1].scatter(targets, errors, alpha=0.5, s=20)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', label='Нет ошибки')
    axes[1, 1].set_xlabel('Реальный STOI', fontsize=12)
    axes[1, 1].set_ylabel('Ошибка (Predicted - True)', fontsize=12)
    axes[1, 1].set_title('Ошибка в зависимости от STOI', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Сохраняем график
    save_path = os.path.join(output_dir, 'test_results.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"График сохранен: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Тестирование модели предсказания STOI')
    parser.add_argument('--checkpoint', type=str, required=True, default='checkpoints/best_model.pt',
                       help='Путь к чекпоинту модели')
    parser.add_argument('--audio_dir', type=str,
                       default='/home/danya/datasets/CMU-MOSEI/Audio/',
                       help='Базовая директория с обработанными аудио файлами (будет искать в поддиректориях noise, reverb, noise_reverb)')
    parser.add_argument('--original_dir', type=str,
                       default='/home/danya/datasets/CMU-MOSEI/Audio/WAV_16000/',
                       help='Директория с оригинальными аудио файлами')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                       help='Директория для сохранения результатов')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Размер батча')
    parser.add_argument('--wav2vec_model', type=str, default='facebook/wav2vec2-base',
                       help='Название модели wav2vec2')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Размерность скрытых слоев')
    parser.add_argument('--num_layers', type=int, default=5,
                       help='Количество полносвязных слоев')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--use_residual', action='store_true', default=True,
                       help='Использовать residual connections')
    parser.add_argument('--freeze_wav2vec', action='store_true', default=True,
                       help='Замораживать веса wav2vec (False = fine-tuning)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Частота дискретизации')
    parser.add_argument('--max_length', type=float, default=10.0,
                       help='Максимальная длина аудио в секундах')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Максимальное количество образцов для тестирования (None = использовать все)')
    parser.add_argument('--single_chunk_per_audio', action='store_true', default=False,
                       help='Брать только один чанк на аудио (для быстрого тестового прогона)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Устройство для тестирования')
    
    args = parser.parse_args()
    
    # Создаем директорию для результатов
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Создаем dataset
    print("Загрузка тестового датасета...")
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
    
    print(f"Тестовых образцов: {len(dataset)}")
    
    # Ограничиваем размер датасета, если указано
    if args.max_samples is not None and args.max_samples > 0:
        if args.max_samples < len(dataset):
            print(f"Ограничиваем тестовый датасет до {args.max_samples} образцов (было {len(dataset)})")
            indices = torch.randperm(len(dataset))[:args.max_samples].tolist()
            dataset = torch.utils.data.Subset(dataset, indices)
        else:
            print(f"max_samples ({args.max_samples}) больше размера датасета ({len(dataset)}), используем все образцы")
    
    # Создаем DataLoader
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True if args.device == 'cuda' else False,
        persistent_workers=True if args.device == 'cuda' else False
    )
    
    # Создаем модель
    print("Загрузка модели...")
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
    
    # Загружаем чекпоинт
    # В PyTorch 2.6+ по умолчанию weights_only=True, но наши чекпоинты содержат numpy скаляры
    # Поскольку это наш собственный чекпоинт, безопасно использовать weights_only=False
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Модель загружена из эпохи {checkpoint.get('epoch', 'unknown')}")
    
    # Тестирование
    print("\nНачало тестирования...")
    results = test_model(model, test_loader, args.device)
    
    # Выводим результаты
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("="*60)
    print(f"Mean Absolute Error (MAE):     {results['mae']:.4f}")
    print(f"Mean Squared Error (MSE):      {results['mse']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {results['rmse']:.4f}")
    print(f"R² Score:                      {results['r2']:.4f}")
    print(f"Correlation:                   {results['correlation']:.4f}")
    print("\nТочность в пределах порога:")
    for threshold, accuracy in results['accuracy_at_threshold'].items():
        print(f"  ±{threshold:.2f}: {accuracy:.2f}%")
    print("="*60)
    
    # Строим графики
    print("\nПостроение графиков...")
    plot_results(results, args.output_dir)
    
    # Сохраняем результаты в файл
    results_file = os.path.join(args.output_dir, 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ\n")
        f.write("="*60 + "\n")
        f.write(f"Mean Absolute Error (MAE):     {results['mae']:.4f}\n")
        f.write(f"Mean Squared Error (MSE):      {results['mse']:.4f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {results['rmse']:.4f}\n")
        f.write(f"R² Score:                      {results['r2']:.4f}\n")
        f.write(f"Correlation:                   {results['correlation']:.4f}\n")
        f.write("\nТочность в пределах порога:\n")
        for threshold, accuracy in results['accuracy_at_threshold'].items():
            f.write(f"  ±{threshold:.2f}: {accuracy:.2f}%\n")
    
    print(f"\nРезультаты сохранены в: {results_file}")
    print("Тестирование завершено!")


if __name__ == '__main__':
    main()

