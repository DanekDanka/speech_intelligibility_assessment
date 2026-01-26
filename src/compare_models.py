"""
Скрипт для сравнения метрик всех моделей на тестовом датасете
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
import json
from datetime import datetime

from dataset import STOIDataset
from model import TransformerSTOIPredictor, LSTMSTOIPredictor, CNNSTOIPredictor, WhisperSTOIPredictor

warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad=True.*")


def test_model(model, dataloader, device, model_name):
    """Тестирование модели"""
    model.eval()
    all_preds = []
    all_targets = []
    
    if device == 'cuda':
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Testing {model_name}')
        for batch_idx, batch in enumerate(pbar):
            waveform = batch['waveform'].to(device)
            stoi_true = batch['stoi'].to(device)
            
            # Forward pass
            stoi_pred = model(waveform)
            
            # Убеждаемся, что размеры правильные
            stoi_pred_flat = stoi_pred.view(-1)
            stoi_true_flat = stoi_true.view(-1)
            
            # Сохраняем результаты
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
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'correlation': correlation,
        'accuracy_at_threshold': accuracy_at_threshold
    }


def plot_comparison(all_results, output_dir):
    """Строит графики сравнения моделей"""
    model_names = list(all_results.keys())
    n_models = len(model_names)
    
    # Создаем фигуру с несколькими графиками
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Scatter plots для каждой модели
    for i, model_name in enumerate(model_names):
        ax = plt.subplot(3, n_models, i + 1)
        result = all_results[model_name]
        preds = result['predictions']
        targets = result['targets']
        
        ax.scatter(targets, preds, alpha=0.5, s=20)
        ax.plot([0, 1], [0, 1], 'r--', label='Идеальная линия')
        ax.set_xlabel('Реальный STOI', fontsize=10)
        ax.set_ylabel('Предсказанный STOI', fontsize=10)
        ax.set_title(f'{model_name}\nR² = {result["r2"]:.4f}, Corr = {result["correlation"]:.4f}', 
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 2. Гистограммы ошибок
    for i, model_name in enumerate(model_names):
        ax = plt.subplot(3, n_models, n_models + i + 1)
        result = all_results[model_name]
        errors = result['predictions'] - result['targets']
        
        ax.hist(errors, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Ошибка (Predicted - True)', fontsize=10)
        ax.set_ylabel('Количество', fontsize=10)
        ax.set_title(f'{model_name}\nMAE = {result["mae"]:.4f}, RMSE = {result["rmse"]:.4f}', 
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='r', linestyle='--')
    
    # 3. Сравнение метрик (bar plot)
    ax = plt.subplot(3, 1, 3)
    metrics = ['MAE', 'RMSE', 'R²', 'Correlation']
    x = np.arange(len(metrics))
    width = 0.8 / n_models
    
    for i, model_name in enumerate(model_names):
        result = all_results[model_name]
        values = [
            result['mae'],
            result['rmse'],
            result['r2'],
            result['correlation']
        ]
        offset = (i - n_models/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model_name, alpha=0.8)
    
    ax.set_xlabel('Метрики', fontsize=12)
    ax.set_ylabel('Значение', fontsize=12)
    ax.set_title('Сравнение метрик моделей', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Сохраняем график
    save_path = os.path.join(output_dir, 'models_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"График сохранен: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Сравнение моделей для предсказания STOI')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints_all_models',
                       help='Директория с чекпоинтами моделей')
    parser.add_argument('--audio_dir', type=str,
                       default='/home/danya/datasets/CMU-MOSEI/Audio/',
                       help='Базовая директория с обработанными аудио файлами (будет искать в поддиректориях noise, reverb, noise_reverb)')
    parser.add_argument('--original_dir', type=str,
                       default='/home/danya/datasets/CMU-MOSEI/Audio/WAV_16000/',
                       help='Директория с оригинальными аудио файлами')
    parser.add_argument('--output_dir', type=str, default='./comparison_results',
                       help='Директория для сохранения результатов')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Размер батча')
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Частота дискретизации')
    parser.add_argument('--max_length', type=float, default=10.0,
                       help='Максимальная длина аудио в секундах')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Максимальное количество образцов для тестирования')
    parser.add_argument('--single_chunk_per_audio', action='store_true', default=True,
                       help='Брать только один чанк на аудио (для быстрого тестового прогона)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Устройство для тестирования')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['cnn'],
                       choices=['transformer', 'lstm', 'cnn', 'whisper'],
                       help='Какие модели тестировать')
    
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
        use_wav2vec=False,
        subdirs=['noise', 'reverb', 'noise_reverb', 'extreme_stoi'],  # Ищем файлы в этих поддиректориях
        single_chunk_per_audio=args.single_chunk_per_audio
    )
    
    print(f"Тестовых образцов: {len(dataset)}")
    
    # Ограничиваем размер датасета, если указано
    if args.max_samples is not None and args.max_samples > 0:
        if args.max_samples < len(dataset):
            print(f"Ограничиваем тестовый датасет до {args.max_samples} образцов")
            indices = torch.randperm(len(dataset))[:args.max_samples].tolist()
            dataset = torch.utils.data.Subset(dataset, indices)
    
    # Создаем DataLoader
    test_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if args.device == 'cuda' else False,
        persistent_workers=True if args.device == 'cuda' else False
    )
    
    # Определяем модели
    models_config = {
        'transformer': {
            'class': TransformerSTOIPredictor,
            'checkpoint': 'best_transformer.pt'
        },
        'lstm': {
            'class': LSTMSTOIPredictor,
            'checkpoint': 'best_lstm.pt'
        },
        'cnn': {
            'class': CNNSTOIPredictor,
            'checkpoint': 'best_cnn.pt'
        },
        'whisper': {
            'class': WhisperSTOIPredictor,
            'checkpoint': 'best_whisper.pt'
        }
    }
    
    # Тестируем модели
    all_results = {}
    
    for model_name in args.models:
        if model_name not in models_config:
            print(f"Пропускаем неизвестную модель: {model_name}")
            continue
        
        config = models_config[model_name]
        checkpoint_path = os.path.join(args.checkpoints_dir, config['checkpoint'])
        
        if not os.path.exists(checkpoint_path):
            print(f"Чекпоинт не найден: {checkpoint_path}")
            continue
        
        print(f"\nЗагрузка модели: {model_name}")
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        
        # Проверяем, что имя модели в чекпоинте совпадает
        checkpoint_model_name = checkpoint.get('model_name', model_name)
        if checkpoint_model_name != model_name:
            print(f"Предупреждение: имя модели в чекпоинте ({checkpoint_model_name}) не совпадает с ожидаемым ({model_name})")
        
        # Создаем модель с параметрами из чекпоинта
        model_kwargs = checkpoint.get('model_kwargs', {})
        if not model_kwargs:
            print(f"Предупреждение: model_kwargs не найдены в чекпоинте, используем значения по умолчанию")
        
        model = config['class'](**model_kwargs).to(args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        epoch = checkpoint.get('epoch', 'unknown')
        val_loss = checkpoint.get('val_loss', 'unknown')
        val_mae = checkpoint.get('val_mae', 'unknown')
        print(f"Модель {model_name} загружена:")
        print(f"  - Эпоха: {epoch}")
        print(f"  - Val Loss: {val_loss:.4f}" if isinstance(val_loss, float) else f"  - Val Loss: {val_loss}")
        print(f"  - Val MAE: {val_mae:.4f}" if isinstance(val_mae, float) else f"  - Val MAE: {val_mae}")
        
        # Тестирование
        result = test_model(model, test_loader, args.device, model_name)
        all_results[model_name] = result
        
        # Выводим результаты
        print(f"\nРезультаты для {model_name}:")
        print(f"  MAE:  {result['mae']:.4f}")
        print(f"  RMSE: {result['rmse']:.4f}")
        print(f"  R²:   {result['r2']:.4f}")
        print(f"  Corr: {result['correlation']:.4f}")
    
    if not all_results:
        print("Нет результатов для сравнения!")
        return
    
    # Строим графики сравнения
    print("\nПостроение графиков сравнения...")
    plot_comparison(all_results, args.output_dir)
    
    # Сохраняем результаты в файл
    comparison_data = {
        'timestamp': datetime.now().isoformat(),
        'test_samples': len(dataset),
        'results': {}
    }
    
    for model_name, result in all_results.items():
        comparison_data['results'][model_name] = {
            'mae': float(result['mae']),
            'mse': float(result['mse']),
            'rmse': float(result['rmse']),
            'r2': float(result['r2']),
            'correlation': float(result['correlation']),
            'accuracy_at_threshold': {str(k): float(v) for k, v in result['accuracy_at_threshold'].items()}
        }
    
    results_file = os.path.join(args.output_dir, 'comparison_metrics.json')
    with open(results_file, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    # Выводим таблицу сравнения
    print("\n" + "="*120)
    print("СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*120)
    print(f"{'Модель':<20} {'MAE':<12} {'RMSE':<12} {'R²':<12} {'Correlation':<12} {'±0.05':<10} {'±0.10':<10} {'±0.15':<10}")
    print("-"*120)
    
    for model_name, result in all_results.items():
        acc_005 = result['accuracy_at_threshold'].get(0.05, 0.0)
        acc_010 = result['accuracy_at_threshold'].get(0.10, 0.0)
        acc_015 = result['accuracy_at_threshold'].get(0.15, 0.0)
        print(f"{model_name:<20} "
              f"{result['mae']:>11.4f} "
              f"{result['rmse']:>11.4f} "
              f"{result['r2']:>11.4f} "
              f"{result['correlation']:>11.4f} "
              f"{acc_005:>9.2f}% "
              f"{acc_010:>9.2f}% "
              f"{acc_015:>9.2f}%")
    
    print("="*120)
    print("\nТочность в пределах порога показывает процент предсказаний с ошибкой ≤ указанного значения")
    print(f"\nРезультаты сохранены в: {results_file}")
    print("Сравнение завершено!")


if __name__ == '__main__':
    main()

