#!/usr/bin/env python3
"""
Скрипт для обучения всех моделей и автоматического сравнения метрик
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Обучение всех моделей и сравнение метрик')
    parser.add_argument('--audio_dir', type=str, 
                       default='/home/danya/datasets/CMU-MOSEI/Audio/',
                       help='Базовая директория с обработанными аудио файлами (будет искать в поддиректориях noise, reverb, noise_reverb, extreme_stoi)')
    parser.add_argument('--original_dir', type=str,
                       default='/home/danya/datasets/CMU-MOSEI/Audio/WAV_16000/',
                       help='Директория с оригинальными аудио файлами')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_all_models',
                       help='Директория для сохранения чекпоинтов')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Размер батча')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Количество эпох')
    parser.add_argument('--lr', type=float, default=5e-4,
                       help='Learning rate (рекомендуется 5e-4 для стабильного обучения)')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['transformer', 'lstm', 'cnn'],
                       choices=['transformer', 'lstm', 'cnn'],
                       help='Какие модели обучать')
    parser.add_argument('--skip_training', action='store_true',
                       help='Пропустить обучение и только сравнить существующие модели')
    parser.add_argument('--comparison_dir', type=str, default='./comparison_results',
                       help='Директория для сохранения результатов сравнения')
    
    args = parser.parse_args()
    
    # Шаг 1: Обучение моделей
    if not args.skip_training:
        print("="*80)
        print("ШАГ 1: ОБУЧЕНИЕ МОДЕЛЕЙ")
        print("="*80)
        
        train_cmd = [
            sys.executable, 'train_all_models.py',
            '--audio_dir', args.audio_dir,
            '--original_dir', args.original_dir,
            '--output_dir', args.output_dir,
            '--batch_size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--lr', str(args.lr),
            '--models'] + args.models
        
        print(f"Запуск команды: {' '.join(train_cmd)}")
        result = subprocess.run(train_cmd, cwd=Path(__file__).parent)
        
        if result.returncode != 0:
            print("Ошибка при обучении моделей!")
            return 1
    else:
        print("Пропуск обучения (используются существующие модели)")
    
    # Шаг 2: Сравнение моделей
    print("\n" + "="*80)
    print("ШАГ 2: СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*80)
    
    compare_cmd = [
        sys.executable, 'compare_models.py',
        '--checkpoints_dir', args.output_dir,
        '--audio_dir', args.audio_dir,
        '--original_dir', args.original_dir,
        '--output_dir', args.comparison_dir,
        '--batch_size', str(args.batch_size),
        '--models'] + args.models
    
    print(f"Запуск команды: {' '.join(compare_cmd)}")
    result = subprocess.run(compare_cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print("Ошибка при сравнении моделей!")
        return 1
    
    print("\n" + "="*80)
    print("ВСЕ ЗАДАЧИ ВЫПОЛНЕНЫ!")
    print("="*80)
    print(f"Результаты сравнения сохранены в: {args.comparison_dir}")
    print(f"Графики сравнения: {args.comparison_dir}/models_comparison.png")
    print(f"Метрики в JSON: {args.comparison_dir}/comparison_metrics.json")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

