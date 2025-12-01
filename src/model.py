"""
Модель для предсказания STOI на основе wav2vec2
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor


class STOIPredictor(nn.Module):
    """
    Модель для предсказания STOI из аудио сигнала.
    
    Использует предобученный wav2vec2 для извлечения признаков из аудио,
    затем добавляет дополнительные признаки (SNR, RT60, wet_level) и
    предсказывает STOI через полносвязные слои.
    """
    
    def __init__(self, 
                 wav2vec_model_name='facebook/wav2vec2-base',
                 hidden_dim=256,
                 num_layers=3,
                 dropout=0.1,
                 use_audio_features=True,
                 use_metadata_features=True):
        """
        Args:
            wav2vec_model_name: Название предобученной модели wav2vec2
            hidden_dim: Размерность скрытых слоев
            num_layers: Количество полносвязных слоев
            dropout: Dropout rate
            use_audio_features: Использовать ли признаки из аудио (wav2vec)
            use_metadata_features: Использовать ли метаданные (SNR, RT60, wet)
        """
        super(STOIPredictor, self).__init__()
        
        self.use_audio_features = use_audio_features
        self.use_metadata_features = use_metadata_features
        
        # Загружаем предобученный wav2vec2
        if use_audio_features:
            # Используем safetensors для безопасной загрузки
            self.wav2vec = Wav2Vec2Model.from_pretrained(
                wav2vec_model_name,
                use_safetensors=True
            )
            # Замораживаем веса wav2vec (опционально, можно разморозить для fine-tuning)
            # for param in self.wav2vec.parameters():
            #     param.requires_grad = False
            
            # Размерность признаков wav2vec2-base: 768
            wav2vec_dim = self.wav2vec.config.hidden_size
        else:
            wav2vec_dim = 0
        
        # Размерность метаданных (SNR, RT60, wet_level)
        metadata_dim = 3 if use_metadata_features else 0
        
        # Общая размерность входных признаков
        input_dim = wav2vec_dim + metadata_dim
        
        # Полносвязные слои для предсказания STOI
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # Используем LayerNorm вместо BatchNorm (не зависит от размера батча)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # Выходной слой (STOI в диапазоне [0, 1])
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())  # Ограничиваем выход в [0, 1]
        
        self.fc_layers = nn.Sequential(*layers)
        
        # Процессор для wav2vec2
        if use_audio_features:
            self.processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_name)
    
    def forward(self, waveform, features=None):
        """
        Forward pass
        
        Args:
            waveform: Аудио сигнал, shape (batch_size, seq_len)
            features: Метаданные (SNR, RT60, wet), shape (batch_size, 3)
        
        Returns:
            Предсказанный STOI, shape (batch_size, 1)
        """
        batch_size = waveform.shape[0]
        
        # Извлекаем признаки из аудио с помощью wav2vec2
        if self.use_audio_features:
            # Нормализуем waveform для wav2vec2
            # wav2vec2 ожидает значения в диапазоне [-1, 1]
            waveform_normalized = waveform / (torch.abs(waveform).max(dim=1, keepdim=True)[0] + 1e-8)
            
            # Получаем признаки из wav2vec2
            # В режиме eval всегда используем no_grad для экономии памяти
            with torch.no_grad() if not self.training else torch.enable_grad():
                outputs = self.wav2vec(waveform_normalized)
                # Используем последний скрытый слой
                audio_features = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
                # Глобальное усреднение по временной оси
                audio_features = audio_features.mean(dim=1)  # (batch_size, hidden_dim)
        else:
            audio_features = torch.zeros(batch_size, 0, device=waveform.device)
        
        # Объединяем признаки
        if self.use_audio_features and self.use_metadata_features:
            combined_features = torch.cat([audio_features, features], dim=1)
        elif self.use_audio_features:
            combined_features = audio_features
        elif self.use_metadata_features:
            combined_features = features
        else:
            raise ValueError("Должен быть включен хотя бы один тип признаков")
        
        # Предсказываем STOI
        stoi_pred = self.fc_layers(combined_features)
        
        return stoi_pred
    
    def predict(self, waveform, features=None):
        """
        Предсказание STOI (режим inference)
        
        Args:
            waveform: Аудио сигнал, shape (batch_size, seq_len) или (seq_len,)
            features: Метаданные, shape (batch_size, 3) или (3,)
        
        Returns:
            Предсказанный STOI
        """
        self.eval()
        with torch.no_grad():
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if features is not None and features.dim() == 1:
                features = features.unsqueeze(0)
            
            pred = self.forward(waveform, features)
            return pred.squeeze().cpu().item() if pred.numel() == 1 else pred.squeeze().cpu()
