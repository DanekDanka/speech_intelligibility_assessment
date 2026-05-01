#!/bin/bash

# Скрипт для запуска демонстрационной программы

# Проверяем, что путь к модели передан аргументом
if [ -z "$1" ]; then
    echo "Ошибка: не указан путь к модели."
    echo "Использование: $0 <путь_к_модели>"
    echo "Пример: $0 checkpoints_src_stoi_net/best.pt"
    echo "Пример HF: $0 hf://DanekDanka/NI-STOI/best.pt"
    exit 1
fi

MODEL_ARG="$1"

if [[ "$MODEL_ARG" == hf://* ]]; then
    HF_REF="${MODEL_ARG#hf://}"
    HF_OWNER="${HF_REF%%/*}"
    HF_REST="${HF_REF#*/}"
    HF_REPO_NAME="${HF_REST%%/*}"
    HF_FILE="${HF_REST#*/}"
    HF_REPO="${HF_OWNER}/${HF_REPO_NAME}"

    if [ -z "$HF_OWNER" ] || [ -z "$HF_REPO_NAME" ] || [ -z "$HF_FILE" ] || [ "$HF_REST" = "$HF_FILE" ]; then
        echo "Ошибка: неверный формат HF-ссылки."
        echo "Ожидается: hf://<owner>/<repo>/<path/to/file>"
        echo "Пример: hf://DanekDanka/NI-STOI/best.pt"
        exit 1
    fi

    echo "Скачивание модели из Hugging Face: $HF_REPO/$HF_FILE"
    MODEL_CHECKPOINT="$(python - "$HF_REPO" "$HF_FILE" <<'PY'
import sys
from huggingface_hub import hf_hub_download

repo_id = sys.argv[1]
filename = sys.argv[2]
path = hf_hub_download(repo_id=repo_id, filename=filename)
print(path)
PY
)"

    if [ $? -ne 0 ] || [ -z "$MODEL_CHECKPOINT" ]; then
        echo "Ошибка: не удалось скачать модель из Hugging Face."
        echo "Убедитесь, что установлен пакет huggingface_hub и файл существует в репозитории."
        exit 1
    fi
else
    MODEL_CHECKPOINT="$MODEL_ARG"
fi

export MODEL_CHECKPOINT

# Проверяем наличие модели
if [ ! -f "$MODEL_CHECKPOINT" ]; then
    echo "Внимание: Модель не найдена по пути: $MODEL_CHECKPOINT"
    echo "Передайте корректный путь к модели первым аргументом."
    echo ""
    echo "Пример:"
    echo "  $0 checkpoints_src_stoi_net/best.pt"
    echo "  $0 hf://DanekDanka/NI-STOI/best.pt"
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
