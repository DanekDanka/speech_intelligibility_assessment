import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from pystoi import stoi
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Конфигурация
class Config:
    DATA_DIR = "/home/danya/develop/datasets/CMU-MOSEI/Audio/balanced_stoi_dataset/"
    METADATA_PATH = os.path.join(DATA_DIR, "metadata.csv")
    AUDIO_DIR = os.path.join(DATA_DIR, "audio")
    BATCH_SIZE = 32
    SAMPLE_RATE = 16000
    DURATION = 3  # seconds (будем обрезать/дополнять до этой длины)
    N_FFT = 512
    HOP_LENGTH = 256
    N_MELS = 64
    TRAIN_RATIO = 0.8
    LEARNING_RATE = 0.001
    EPOCHS = 50
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Датасет
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
        
        # Загрузка аудио
        audio, sr = sf.read(audio_path)
        
        # Ресемплинг, если нужно
        if sr != self.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
        
        # Нормализация
        audio = audio / np.max(np.abs(audio))
        
        # Обрезка/дополнение до фиксированной длины
        if len(audio) > self.n_samples:
            audio = audio[:self.n_samples]
        else:
            padding = self.n_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        
        # Извлечение мел-спектрограммы
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Нормализация спектрограммы
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        # Преобразование в тензор
        mel_spec_db = torch.FloatTensor(mel_spec_db)
        stoi_score = torch.FloatTensor([stoi_score])
        
        return mel_spec_db, stoi_score

# Модель CNN для регрессии STOI
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
        
        # Вычисление размера после сверточных слоев
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
        x = x.unsqueeze(1)  # Добавляем размерность канала
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Выравнивание
        x = self.fc_layers(x)
        return x

# Функции для обучения и оценки
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss, np.array(all_targets), np.array(all_predictions)

def plot_results(targets, predictions, epoch, phase):
    plt.figure(figsize=(10, 6))
    plt.scatter(targets, predictions, alpha=0.3)
    plt.plot([0, 1], [0, 1], 'r--')  # Линия идеального предсказания
    plt.xlabel('True STOI')
    plt.ylabel('Predicted STOI')
    plt.title(f'{phase} - Epoch {epoch}: True vs Predicted STOI')
    plt.grid(True)
    
    # Вычисление метрик
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    
    plt.text(0.05, 0.9, f'RMSE: {rmse:.4f}\nR2: {r2:.4f}', 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/{phase}_epoch_{epoch}.png")
    plt.close()

# Основная функция
def main():
    # Загрузка данных
    df = pd.read_csv(Config.METADATA_PATH)
    
    # Разделение на train/test
    train_size = int(Config.TRAIN_RATIO * len(df))
    test_size = len(df) - train_size
    train_df, test_df = random_split(df, [train_size, test_size])
    train_df = df.iloc[train_df.indices]
    test_df = df.iloc[test_df.indices]
    
    # Создание датасетов и загрузчиков
    train_dataset = STOIDataset(
        train_df, Config.AUDIO_DIR, Config.SAMPLE_RATE, 
        Config.DURATION, Config.N_FFT, Config.HOP_LENGTH, Config.N_MELS
    )
    test_dataset = STOIDataset(
        test_df, Config.AUDIO_DIR, Config.SAMPLE_RATE, 
        Config.DURATION, Config.N_FFT, Config.HOP_LENGTH, Config.N_MELS
    )
    
    # Проверка формы данных
    sample_mel, _ = train_dataset[0]
    input_shape = sample_mel.shape
    
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4
    )
    
    # Инициализация модели
    model = STOIPredictor(input_shape).to(Config.DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Обучение
    best_loss = float('inf')
    train_losses = []
    test_losses = []
    
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        
        # Обучение
        train_loss = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        train_losses.append(train_loss)
        
        # Оценка
        test_loss, test_targets, test_predictions = evaluate(model, test_loader, criterion, Config.DEVICE)
        test_losses.append(test_loss)
        
        # Визуализация
        plot_results(test_targets, test_predictions, epoch + 1, "test")
        
        # Печать результатов
        print(f"Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")
        
        # Сохранение лучшей модели
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model")
        
        # Обновление learning rate
        scheduler.step(test_loss)
    
    # График обучения
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig("results/training_curve.png")
    plt.close()
    
    # Финальная оценка
    print("\nFinal Evaluation:")
    model.load_state_dict(torch.load("best_model.pth"))
    test_loss, test_targets, test_predictions = evaluate(model, test_loader, criterion, Config.DEVICE)
    
    rmse = np.sqrt(mean_squared_error(test_targets, test_predictions))
    r2 = r2_score(test_targets, test_predictions)
    
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R2 Score: {r2:.4f}")
    
    # Сохранение модели
    torch.save(model.state_dict(), "final_model.pth")
    print("\nTraining completed and model saved!")

if __name__ == "__main__":
    main()
