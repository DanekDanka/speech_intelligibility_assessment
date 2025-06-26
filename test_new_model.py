import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Конфигурация (должна совпадать с конфигурацией обучения)
class Config:
    DATA_DIR = "/home/danya/develop/datasets/CMU-MOSEI/Audio/balanced_stoi_dataset/"
    METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")
    AUDIO_DIR = os.path.join(DATA_DIR, "audio")
    BATCH_SIZE = 32
    SAMPLE_RATE = 16000
    DURATION = 3  # seconds
    N_FFT = 512
    HOP_LENGTH = 256
    N_MELS = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "best_model.pth"  # путь к сохраненной модели

# Датасет (должен совпадать с тем, что использовался при обучении)
class STOIDataset(Dataset):
    def __init__(self, df, audio_dir, sample_rate, duration, n_fft, hop_length, n_mels):
        self.df = df
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = sample_rate * duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.df.iloc[idx]['filename'])
        stoi_score = self.df.iloc[idx]['stoi']
        
        audio, sr = sf.read(audio_path)
        
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        audio = audio / np.max(np.abs(audio))
        
        if len(audio) > self.n_samples:
            audio = audio[:self.n_samples]
        else:
            padding = self.n_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        mel_spec_db = torch.FloatTensor(mel_spec_db)
        stoi_score = torch.FloatTensor([stoi_score])
        
        return mel_spec_db, stoi_score

# Модель (должна совпадать с архитектурой при обучении)
class STOIPredictor(nn.Module):
    def __init__(self, input_shape):
        super(STOIPredictor, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_shape)
            dummy_output = self.conv_layers(dummy_input)
            self.flatten_size = dummy_output.numel()
        
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def evaluate_model(model, dataloader, device):
    model.eval()
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())
    
    return np.array(all_targets), np.array(all_predictions)

def plot_results(targets, predictions, model_name):
    # Создаем папку для результатов, если её нет
    os.makedirs("eval_results", exist_ok=True)
    
    # График предсказаний vs истинных значений
    plt.figure(figsize=(10, 8))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # Линия идеальных предсказаний
    plt.xlabel('True STOI')
    plt.ylabel('Predicted STOI')
    plt.title(f'True vs Predicted STOI\n{model_name}')
    plt.grid(True)
    
    # Вычисление и вывод метрик
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    mae = np.mean(np.abs(targets - predictions))
    
    metrics_text = f'RMSE: {rmse:.4f}\nR2: {r2:.4f}\nMAE: {mae:.4f}'
    plt.text(0.05, 0.85, metrics_text, transform=plt.gca().transAxes, 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.savefig(f"eval_results/{model_name}_scatter.png")
    plt.close()
    
    # График распределения ошибок
    errors = targets - predictions
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7)
    plt.xlabel('Prediction Error (True - Predicted)')
    plt.ylabel('Count')
    plt.title(f'Prediction Error Distribution\n{model_name}')
    plt.grid(True)
    plt.savefig(f"eval_results/{model_name}_error_dist.png")
    plt.close()
    
    # График распределения STOI
    plt.figure(figsize=(10, 6))
    plt.hist(targets, bins=20, alpha=0.7, label='True')
    plt.hist(predictions, bins=20, alpha=0.7, label='Predicted')
    plt.xlabel('STOI Value')
    plt.ylabel('Count')
    plt.title(f'STOI Value Distribution\n{model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"eval_results/{model_name}_stoi_dist.png")
    plt.close()
    
    return rmse, r2, mae

def load_and_evaluate():
    # Загрузка данных
    df = pd.read_csv(Config.METADATA_PATH)
    
    # Для теста используем 20% данных (или можно загрузить отдельный test.csv)
    test_df = df.sample(frac=0.2, random_state=42)
    
    # Создание датасета и загрузчика
    test_dataset = STOIDataset(
        test_df, Config.AUDIO_DIR, Config.SAMPLE_RATE, 
        Config.DURATION, Config.N_FFT, Config.HOP_LENGTH, Config.N_MELS
    )
    
    # Проверка формы входных данных
    sample_mel, _ = test_dataset[0]
    input_shape = sample_mel.shape
    
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4
    )
    
    # Инициализация модели
    model = STOIPredictor(input_shape).to(Config.DEVICE)
    
    # Загрузка весов
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH))
        print(f"Model loaded from {Config.MODEL_PATH}")
    else:
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")

    print(model)
    
    # Оценка модели
    targets, predictions = evaluate_model(model, test_loader, Config.DEVICE)
    
    # Построение графиков и расчет метрик
    model_name = os.path.splitext(os.path.basename(Config.MODEL_PATH))[0]
    rmse, r2, mae = plot_results(targets, predictions, model_name)
    
    print("\nEvaluation Results:")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Сохранение предсказаний для дальнейшего анализа
    results_df = pd.DataFrame({
        'filename': test_df['filename'].values[:len(targets)],
        'true_stoi': targets.flatten(),
        'predicted_stoi': predictions.flatten(),
        'error': (targets - predictions).flatten()
    })
    results_df.to_csv(f"eval_results/{model_name}_predictions.csv", index=False)
    
    print(f"\nResults saved in 'eval_results' directory")

if __name__ == "__main__":
    load_and_evaluate()
