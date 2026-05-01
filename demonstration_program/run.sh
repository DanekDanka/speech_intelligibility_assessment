#!/bin/bash

# Скрипт для запуска демонстрационной программы

# Проверяем, что путь к модели передан аргументом
if [ -z "$1" ]; then
    echo "Ошибка: не указан путь к модели."
    echo "Использование: $0 <путь_к_модели>"
    echo "Пример: $0 checkpoints_src_stoi_net/best.pt"
    exit 1
fi

export MODEL_CHECKPOINT="$1"

# Проверяем наличие модели
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "Внимание: Модель не найдена по пути: $MODEL_CHECKPOINT"
    echo "Передайте корректный путь к модели первым аргументом."
    echo ""
    echo "Пример:"
    echo "  $0 checkpoints_src_stoi_net/best.pt"
    echo ""
    read -p "Продолжить все равно? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Запускаем приложение
echo "Запуск демонстрационной программы..."
echo "Модель: $MODEL_CHECKPOINT"
echo ""
python app.py
