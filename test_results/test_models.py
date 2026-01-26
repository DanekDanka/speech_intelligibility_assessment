"""
Скрипт для тестирования моделей и сохранения графиков анализа.
Сохраняет:
  - метрики и графики распределений STOI/ошибок
  - график истории обучения (если есть в чекпоинте)
"""
import os
import json
import argparse
from datetime import datetime
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import STOIDataset
from src.model import TransformerSTOIPredictor, LSTMSTOIPredictor, CNNSTOIPredictor, WhisperSTOIPredictor


def compute_bin_errors(preds, targets, thresholds):
    """
    Вычисляет процент предсказаний, которые попадают в заданные пороги ошибок.
    
    Args:
        preds: массив предсказаний
        targets: массив истинных значений
        thresholds: список порогов ошибок (например, [0.05, 0.10, 0.15, 0.20])
    
    Returns:
        словарь с процентами для каждого порога
    """
    errors = np.abs(preds - targets)
    bin_errors = {}
    
    for threshold in thresholds:
        # Процент предсказаний с ошибкой <= threshold
        accuracy = np.mean(errors <= threshold) * 100.0
        bin_errors[f"bin_error_{threshold:.2f}"] = float(accuracy)
    
    return bin_errors


def test_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Testing")
        for batch in pbar:
            waveform = batch["waveform"].to(device)
            stoi_true = batch["stoi"].to(device)
            stoi_pred = model(waveform)

            preds = stoi_pred.view(-1).cpu().numpy()
            targets = stoi_true.view(-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    mae = mean_absolute_error(all_targets, all_preds)
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)
    correlation = np.corrcoef(all_preds, all_targets)[0, 1]
    
    # Вычисляем коэффициент ранговой корреляции Спирмена (SCC)
    scc, scc_pvalue = spearmanr(all_preds, all_targets)
    
    # Вычисляем Bin Errors для порогов 0.05, 0.10, 0.15, 0.20
    bin_error_thresholds = [0.05, 0.10, 0.15, 0.20]
    bin_errors = compute_bin_errors(all_preds, all_targets, bin_error_thresholds)

    return {
        "predictions": all_preds,
        "targets": all_targets,
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "correlation": float(correlation),
        "scc": float(scc),
        "scc_pvalue": float(scc_pvalue),
        **bin_errors,  # Распаковываем bin_errors в словарь
    }


def plot_training_history(history, output_path):
    if not history:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="train_loss")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="val_loss")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.set_title("Training History")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_prediction_distributions(preds, targets, output_path_prefix):
    # Фильтруем точки вне допустимого диапазона STOI для графиков
    mask = (preds >= 0.0) & (preds <= 1.0) & (targets >= 0.0) & (targets <= 1.0)
    preds = preds[mask]
    targets = targets[mask]

    # Распределение истинных и предсказанных значений
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(targets, bins=30, alpha=0.6, label="True STOI", edgecolor="black")
    ax.hist(preds, bins=30, alpha=0.6, label="Pred STOI", edgecolor="black")
    ax.set_xlabel("STOI")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of True vs Pred STOI")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_stoi_distributions.png", dpi=200)
    plt.close()

    # Распределение ошибок
    errors = preds - targets
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.hist(errors, bins=40, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Error (Pred - True)")
    ax.set_ylabel("Count")
    ax.set_title("Error Distribution")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_error_distribution.png", dpi=200)
    plt.close()

    # Scatter plot: предсказания vs истинные
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    ax.scatter(targets, preds, s=12, alpha=0.5)
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_xlabel("True STOI")
    ax.set_ylabel("Pred STOI")
    ax.set_title("Pred vs True STOI")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_path_prefix}_scatter.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Тестирование моделей и графики анализа")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints_cnn_final")
    parser.add_argument("--audio_dir", type=str, default="/home/danya/datasets/CMU-MOSEI/Audio/")
    parser.add_argument("--original_dir", type=str, default="/home/danya/datasets/CMU-MOSEI/Audio/WAV_16000/")
    parser.add_argument("--output_dir", type=str, default="./test_results")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--max_length", type=float, default=5.0)
    parser.add_argument("--single_chunk_per_audio", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--models", type=str, nargs="+", default=["cnn_hyperopt"],
                        choices=["transformer", "lstm", "cnn", "cnn_hyperopt", "whisper"])
    parser.add_argument("--cnn_hyperopt_checkpoint_dir", type=str, default="./checkpoints_cnn_final",
                        help="Директория с чекпоинтом CNN модели с подобранными гиперпараметрами")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = STOIDataset(
        audio_dir=args.audio_dir,
        original_dir=args.original_dir,
        sample_rate=args.sample_rate,
        max_length_seconds=args.max_length,
        use_wav2vec=False,
        subdirs=["noise", "reverb", "noise_reverb", "extreme_stoi"],
        single_chunk_per_audio=args.single_chunk_per_audio,
    )
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check paths.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if args.device == "cuda" else False,
        persistent_workers=True if args.device == "cuda" else False,
    )

    models_config = {
        "transformer": {"class": TransformerSTOIPredictor, "checkpoint": "best_transformer.pt"},
        "lstm": {"class": LSTMSTOIPredictor, "checkpoint": "best_lstm.pt"},
        "cnn": {"class": CNNSTOIPredictor, "checkpoint": "best_cnn.pt"},
        "cnn_hyperopt": {
            "class": CNNSTOIPredictor, 
            "checkpoint": "best_cnn_final.pt",
            "checkpoint_dir": args.cnn_hyperopt_checkpoint_dir
        },
        "whisper": {"class": WhisperSTOIPredictor, "checkpoint": "best_whisper.pt"},
    }

    results = {
        "timestamp": datetime.now().isoformat(),
        "dataset_size": len(dataset),
        "models": {},
    }

    for model_name in args.models:
        config = models_config[model_name]
        
        # Определяем директорию для чекпоинта
        if "checkpoint_dir" in config:
            checkpoint_dir = config["checkpoint_dir"]
        else:
            checkpoint_dir = args.checkpoints_dir
        
        checkpoint_path = os.path.join(checkpoint_dir, config["checkpoint"])
        if not os.path.exists(checkpoint_path):
            print(f"Чекпоинт не найден: {checkpoint_path}")
            continue

        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        model_kwargs = checkpoint.get("model_kwargs", {})
        
        # Если model_kwargs пустые, пытаемся загрузить из best_hyperparameters.json
        if not model_kwargs and model_name == "cnn_hyperopt":
            hyperparams_file = os.path.join(checkpoint_dir, "best_hyperparameters.json")
            if os.path.exists(hyperparams_file):
                with open(hyperparams_file, 'r') as f:
                    hyperparams_data = json.load(f)
                    model_kwargs = hyperparams_data["hyperparameters"]["model_kwargs"]
        
        model = config["class"](**model_kwargs).to(args.device)
        model.load_state_dict(checkpoint["model_state_dict"])

        print(f"\nTesting model: {model_name}")
        metrics = test_model(model, loader, args.device)

        model_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        plot_prediction_distributions(
            metrics["predictions"],
            metrics["targets"],
            os.path.join(model_dir, model_name),
        )

        history = checkpoint.get("history")
        if history:
            plot_training_history(history, os.path.join(model_dir, f"{model_name}_history.png"))

        results["models"][model_name] = {
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "correlation": metrics["correlation"],
            "scc": metrics["scc"],
            "scc_pvalue": metrics["scc_pvalue"],
            "bin_error_0.05": metrics["bin_error_0.05"],
            "bin_error_0.10": metrics["bin_error_0.10"],
            "bin_error_0.15": metrics["bin_error_0.15"],
            "bin_error_0.20": metrics["bin_error_0.20"],
        }

    results_path = os.path.join(args.output_dir, "summary_metrics.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nРезультаты сохранены в: {args.output_dir}")


if __name__ == "__main__":
    main()

