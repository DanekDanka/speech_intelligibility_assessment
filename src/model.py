"""
Модели для предсказания STOI
"""
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor, WhisperModel, WhisperFeatureExtractor
import math


class STOIPredictor(nn.Module):
    """
    Модель для предсказания STOI из аудио сигнала.
    
    Использует предобученный wav2vec2 для извлечения признаков из аудио,
    затем добавляет дополнительные признаки (SNR, RT60, wet_level) и
    предсказывает STOI через полносвязные слои с residual connections.
    """
    
    def __init__(self, 
                 wav2vec_model_name='facebook/wav2vec2-base',
                 hidden_dim=512,
                 num_layers=5,
                 dropout=0.2,
                 use_audio_features=True,
                 use_metadata_features=True,
                 use_residual=True,
                 freeze_wav2vec=True):
        """
        Args:
            wav2vec_model_name: Название предобученной модели wav2vec2
            hidden_dim: Размерность скрытых слоев
            num_layers: Количество полносвязных слоев
            dropout: Dropout rate
            use_audio_features: Использовать ли признаки из аудио (wav2vec)
            use_metadata_features: Использовать ли метаданные (SNR, RT60, wet)
            use_residual: Использовать ли residual connections
            freeze_wav2vec: Замораживать ли веса wav2vec (False = fine-tuning)
        """
        super(STOIPredictor, self).__init__()
        
        self.use_audio_features = use_audio_features
        self.use_metadata_features = use_metadata_features
        self.use_residual = use_residual
        self.freeze_wav2vec = freeze_wav2vec
        
        # Загружаем предобученный wav2vec2
        if use_audio_features:
            # Используем safetensors для безопасной загрузки
            # Отключаем предупреждение о gradient_checkpointing
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*gradient_checkpointing.*")
                warnings.filterwarnings("ignore", category=FutureWarning, message=".*gradient_checkpointing.*")
                self.wav2vec = Wav2Vec2Model.from_pretrained(
                    wav2vec_model_name,
                    use_safetensors=True
                )
            # Замораживаем веса wav2vec (можно разморозить для fine-tuning)
            if self.freeze_wav2vec:
                for param in self.wav2vec.parameters():
                    param.requires_grad = False
            
            # Размерность признаков wav2vec2-base: 768
            wav2vec_dim = self.wav2vec.config.hidden_size
        else:
            wav2vec_dim = 0
        
        # Размерность метаданных (SNR, RT60, wet_level)
        metadata_dim = 3 if use_metadata_features else 0
        
        # Общая размерность входных признаков
        input_dim = wav2vec_dim + metadata_dim
        
        # Проекция входных признаков до hidden_dim
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Полносвязные слои для предсказания STOI с residual connections
        self.fc_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.fc_layers.append(layer)
        
        # Выходной слой (STOI в диапазоне [0, 1])
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Ограничиваем выход в [0, 1]
        )
        
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
            # Если wav2vec заморожен, используем inference_mode для экономии памяти
            # Если разморожен (fine-tuning), используем обычный режим
            if self.freeze_wav2vec:
                # Используем inference_mode для замороженной модели (более эффективно, чем no_grad)
                with torch.inference_mode():
                    outputs = self.wav2vec(waveform_normalized)
            elif not self.training:
                # В режиме eval используем no_grad
                with torch.no_grad():
                    outputs = self.wav2vec(waveform_normalized)
            else:
                # В режиме обучения с fine-tuning используем обычный режим
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
        
        # Проекция входных признаков
        x = self.input_projection(combined_features)
        
        # Применяем полносвязные слои с residual connections
        for layer in self.fc_layers:
            if self.use_residual:
                x = x + layer(x)  # Residual connection
            else:
                x = layer(x)
        
        # Предсказываем STOI
        stoi_pred = self.output_layer(x)
        
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


class TransformerSTOIPredictor(nn.Module):
    """
    Модель для предсказания STOI на основе Transformer энкодера.
    
    Использует Transformer энкодер для обработки аудио сигнала,
    затем добавляет метаданные и предсказывает STOI.
    """
    
    def __init__(self,
                 input_dim=1,  # Размерность входного аудио (моно = 1)
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=1024,
                 dropout=0.1,
                 max_seq_len=160000,
                 use_metadata_features=True,
                 hidden_dim=512,
                 num_fc_layers=3):
        """
        Args:
            input_dim: Размерность входного аудио (1 для моно)
            d_model: Размерность модели Transformer
            nhead: Количество голов внимания
            num_layers: Количество слоев Transformer
            dim_feedforward: Размерность feedforward сети
            dropout: Dropout rate
            max_seq_len: Максимальная длина последовательности
            use_metadata_features: Использовать ли метаданные
            hidden_dim: Размерность скрытых слоев для финальной сети
            num_fc_layers: Количество полносвязных слоев
        """
        super(TransformerSTOIPredictor, self).__init__()
        
        self.d_model = d_model
        self.use_metadata_features = use_metadata_features
        
        # Downsampling для уменьшения длины последовательности (экономия памяти)
        # Используем свёрточный слой для уменьшения в 4 раза
        self.downsample = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=4,
            stride=4,
            padding=0
        )
        
        # Проекция входного аудио в d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Позиционное кодирование
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer энкодер
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Размерность метаданных
        metadata_dim = 3 if use_metadata_features else 0
        
        # Объединение признаков
        combined_dim = d_model + metadata_dim
        
        # Полносвязные слои для предсказания
        fc_layers = []
        fc_layers.append(nn.Linear(combined_dim, hidden_dim))
        fc_layers.append(nn.LayerNorm(hidden_dim))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(dropout))
        
        for _ in range(num_fc_layers - 1):
            fc_layers.append(nn.Linear(hidden_dim, hidden_dim))
            fc_layers.append(nn.LayerNorm(hidden_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # Выходной слой
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, waveform, features=None):
        """
        Args:
            waveform: Аудио сигнал, shape (batch_size, seq_len)
            features: Метаданные (SNR, RT60, wet), shape (batch_size, 3)
        
        Returns:
            Предсказанный STOI, shape (batch_size, 1)
        """
        batch_size, seq_len = waveform.shape
        
        # Downsampling для уменьшения длины последовательности
        waveform = waveform.unsqueeze(1)  # (batch_size, 1, seq_len)
        waveform = self.downsample(waveform)  # (batch_size, 1, seq_len/4)
        waveform = waveform.squeeze(1)  # (batch_size, seq_len/4)
        
        # Добавляем dimension для input_dim
        waveform = waveform.unsqueeze(-1)  # (batch_size, seq_len/4, 1)
        
        # Убеждаемся, что данные contiguous
        waveform = waveform.contiguous()
        
        # Проекция в d_model
        x = self.input_projection(waveform)  # (batch_size, seq_len/4, d_model)
        
        # Позиционное кодирование
        x = self.pos_encoder(x)
        
        # Убеждаемся, что данные contiguous перед Transformer
        x = x.contiguous()
        
        # Transformer энкодер
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # Глобальное усреднение по временной оси
        audio_features = x.mean(dim=1)  # (batch_size, d_model)
        
        # Объединяем с метаданными
        if self.use_metadata_features and features is not None:
            combined_features = torch.cat([audio_features, features], dim=1)
        else:
            combined_features = audio_features
        
        # Полносвязные слои
        x = self.fc_layers(combined_features)
        
        # Предсказание STOI
        stoi_pred = self.output_layer(x)
        
        return stoi_pred


class PositionalEncoding(nn.Module):
    """Позиционное кодирование для Transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LSTMSTOIPredictor(nn.Module):
    """
    Модель для предсказания STOI на основе LSTM.
    
    Использует двунаправленный LSTM для обработки аудио сигнала,
    затем добавляет метаданные и предсказывает STOI.
    """
    
    def __init__(self,
                 input_dim=1,
                 hidden_dim=256,
                 num_layers=3,
                 dropout=0.2,
                 bidirectional=True,
                 use_metadata_features=True,
                 fc_hidden_dim=512,
                 num_fc_layers=3):
        """
        Args:
            input_dim: Размерность входного аудио (1 для моно)
            hidden_dim: Размерность скрытого состояния LSTM
            num_layers: Количество слоев LSTM
            dropout: Dropout rate
            bidirectional: Использовать ли двунаправленный LSTM
            use_metadata_features: Использовать ли метаданные
            fc_hidden_dim: Размерность скрытых слоев для финальной сети
            num_fc_layers: Количество полносвязных слоев
        """
        super(LSTMSTOIPredictor, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_metadata_features = use_metadata_features
        
        # Downsampling для уменьшения длины последовательности (экономия памяти и стабильность cuDNN)
        # Используем свёрточный слой для уменьшения в 4 раза
        self.downsample = nn.Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=4,
            stride=4,
            padding=0
        )
        
        # LSTM слой
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Размерность выхода LSTM
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Размерность метаданных
        metadata_dim = 3 if use_metadata_features else 0
        
        # Объединение признаков
        combined_dim = lstm_output_dim + metadata_dim
        
        # Полносвязные слои
        fc_layers = []
        fc_layers.append(nn.Linear(combined_dim, fc_hidden_dim))
        fc_layers.append(nn.LayerNorm(fc_hidden_dim))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(dropout))
        
        for _ in range(num_fc_layers - 1):
            fc_layers.append(nn.Linear(fc_hidden_dim, fc_hidden_dim))
            fc_layers.append(nn.LayerNorm(fc_hidden_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # Выходной слой
        self.output_layer = nn.Sequential(
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, waveform, features=None):
        """
        Args:
            waveform: Аудио сигнал, shape (batch_size, seq_len)
            features: Метаданные (SNR, RT60, wet), shape (batch_size, 3)
        
        Returns:
            Предсказанный STOI, shape (batch_size, 1)
        """
        # Убеждаемся, что входные данные contiguous с самого начала
        waveform = waveform.contiguous()
        
        batch_size, seq_len = waveform.shape
        
        # Downsampling для уменьшения длины последовательности
        waveform = waveform.unsqueeze(1)  # (batch_size, 1, seq_len)
        waveform = self.downsample(waveform)  # (batch_size, 1, seq_len/4)
        waveform = waveform.squeeze(1)  # (batch_size, seq_len/4)
        
        # Убеждаемся, что данные contiguous после downsampling
        waveform = waveform.contiguous()
        
        # Добавляем dimension для input_dim
        waveform = waveform.unsqueeze(-1)  # (batch_size, seq_len/4, 1)
        
        # Критически важно: данные должны быть contiguous для cuDNN LSTM
        waveform = waveform.contiguous()
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(waveform)
        
        # Используем последний выходной вектор
        # Для двунаправленного LSTM объединяем forward и backward
        if self.bidirectional:
            forward_hidden = h_n[-2]  # Последний forward hidden state
            backward_hidden = h_n[-1]  # Последний backward hidden state
            audio_features = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            audio_features = h_n[-1]  # Последний hidden state
        
        # Объединяем с метаданными
        if self.use_metadata_features and features is not None:
            combined_features = torch.cat([audio_features, features], dim=1)
        else:
            combined_features = audio_features
        
        # Полносвязные слои
        x = self.fc_layers(combined_features)
        
        # Предсказание STOI
        stoi_pred = self.output_layer(x)
        
        return stoi_pred


class CNNSTOIPredictor(nn.Module):
    """
    Модель для предсказания STOI на основе свёрточной нейронной сети.
    
    Использует 1D свёрточные слои для обработки аудио сигнала,
    затем добавляет метаданные и предсказывает STOI.
    """
    
    def __init__(self,
                 input_dim=1,
                 num_filters=[64, 128, 256, 512],
                 kernel_sizes=[7, 5, 3, 3],
                 stride=2,
                 dropout=0.2,
                 use_metadata_features=True,
                 fc_hidden_dim=512,
                 num_fc_layers=3):
        """
        Args:
            input_dim: Размерность входного аудио (1 для моно)
            num_filters: Список количества фильтров для каждого свёрточного слоя
            kernel_sizes: Список размеров ядер для каждого свёрточного слоя
            stride: Шаг свёртки
            dropout: Dropout rate
            use_metadata_features: Использовать ли метаданные
            fc_hidden_dim: Размерность скрытых слоев для финальной сети
            num_fc_layers: Количество полносвязных слоев
        """
        super(CNNSTOIPredictor, self).__init__()
        
        self.use_metadata_features = use_metadata_features
        
        # Свёрточные слои
        conv_layers = []
        in_channels = input_dim
        
        for i, (out_channels, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            conv_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size//2))
            conv_layers.append(nn.BatchNorm1d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Глобальное усреднение и максимизация
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Размерность признаков после свёрток (сумма avg и max pooling)
        cnn_output_dim = num_filters[-1] * 2
        
        # Размерность метаданных
        metadata_dim = 3 if use_metadata_features else 0
        
        # Объединение признаков
        combined_dim = cnn_output_dim + metadata_dim
        
        # Полносвязные слои
        fc_layers = []
        fc_layers.append(nn.Linear(combined_dim, fc_hidden_dim))
        fc_layers.append(nn.LayerNorm(fc_hidden_dim))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(dropout))
        
        for _ in range(num_fc_layers - 1):
            fc_layers.append(nn.Linear(fc_hidden_dim, fc_hidden_dim))
            fc_layers.append(nn.LayerNorm(fc_hidden_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
        # Выходной слой
        self.output_layer = nn.Sequential(
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, waveform, features=None):
        """
        Args:
            waveform: Аудио сигнал, shape (batch_size, seq_len)
            features: Метаданные (SNR, RT60, wet), shape (batch_size, 3)
        
        Returns:
            Предсказанный STOI, shape (batch_size, 1)
        """
        batch_size, seq_len = waveform.shape
        
        # Добавляем dimension для каналов (batch_size, channels, seq_len)
        waveform = waveform.unsqueeze(1)  # (batch_size, 1, seq_len)
        
        # Убеждаемся, что данные contiguous для cuDNN
        waveform = waveform.contiguous()
        
        # Свёрточные слои
        x = self.conv_layers(waveform)  # (batch_size, num_filters[-1], reduced_seq_len)
        
        # Глобальное усреднение и максимизация
        avg_pool = self.global_pool(x).squeeze(-1)  # (batch_size, num_filters[-1])
        max_pool = self.global_max_pool(x).squeeze(-1)  # (batch_size, num_filters[-1])
        
        # Объединяем avg и max pooling
        audio_features = torch.cat([avg_pool, max_pool], dim=1)  # (batch_size, num_filters[-1] * 2)
        
        # Объединяем с метаданными
        if self.use_metadata_features and features is not None:
            combined_features = torch.cat([audio_features, features], dim=1)
        else:
            combined_features = audio_features
        
        # Полносвязные слои
        x = self.fc_layers(combined_features)
        
        # Предсказание STOI
        stoi_pred = self.output_layer(x)
        
        return stoi_pred


class WhisperSTOIPredictor(nn.Module):
    """
    Модель для предсказания STOI на основе Whisper энкодера.
    
    Использует предобученный Whisper энкодер для извлечения признаков из аудио,
    затем предсказывает STOI через регрессионную голову.
    """
    
    def __init__(self,
                 whisper_model_name='openai/whisper-base',
                 hidden_dim=512,
                 num_fc_layers=5,
                 dropout=0.2,
                 use_metadata_features=False,
                 freeze_encoder=True,
                 use_residual=True):
        """
        Args:
            whisper_model_name: Название предобученной модели Whisper
            hidden_dim: Размерность скрытых слоев для регрессионной головы
            num_fc_layers: Количество полносвязных слоев в регрессионной голове
            dropout: Dropout rate
            use_metadata_features: Использовать ли метаданные (SNR, RT60, wet)
            freeze_encoder: Замораживать ли веса Whisper энкодера
            use_residual: Использовать ли residual connections в полносвязных слоях
        """
        super(WhisperSTOIPredictor, self).__init__()
        
        self.use_metadata_features = use_metadata_features
        self.freeze_encoder = freeze_encoder
        self.use_residual = use_residual
        
        # Загружаем предобученный Whisper модель
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            whisper_model = WhisperModel.from_pretrained(whisper_model_name)
        
        # Используем только энкодер
        self.whisper_encoder = whisper_model.encoder
        
        # Замораживаем веса энкодера, если нужно
        if self.freeze_encoder:
            for param in self.whisper_encoder.parameters():
                param.requires_grad = False
        
        # Размерность признаков Whisper энкодера
        encoder_dim = whisper_model.config.d_model
        
        # Feature extractor для предобработки аудио
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)
        
        # Mel-spectrogram transform для преобразования waveform в mel-spectrogram
        # Whisper использует: n_mels=80, hop_length=160, n_fft=400, sample_rate=16000
        import torchaudio.transforms as T
        self.mel_transform = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=80,
            f_min=0.0,
            f_max=8000.0,
            normalized=False  # Whisper нормализует сам
        )
        
        # Размерность метаданных
        metadata_dim = 3 if use_metadata_features else 0
        
        # Объединение признаков
        combined_dim = encoder_dim + metadata_dim
        
        # Проекция входных признаков до hidden_dim
        self.input_projection = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Улучшенные полносвязные слои с residual connections
        self.fc_layers = nn.ModuleList()
        for i in range(num_fc_layers):
            layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.fc_layers.append(layer)
        
        # Выходной слой с улучшенной архитектурой (3 слоя вместо 2)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Ограничиваем выход в [0, 1]
        )
    
    def forward(self, waveform, features=None):
        """
        Args:
            waveform: Аудио сигнал, shape (batch_size, seq_len)
            features: Метаданные (SNR, RT60, wet), shape (batch_size, 3)
        
        Returns:
            Предсказанный STOI, shape (batch_size, 1)
        """
        batch_size = waveform.shape[0]
        device = waveform.device
        
        # Преобразуем в mel-spectrogram
        # waveform shape: (batch_size, seq_len)
        mel_spec = self.mel_transform(waveform)  # (batch_size, n_mels, time_frames)
        
        # Логарифмируем (как в Whisper) - используем более стабильный способ
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-10))
        
        # Нормализуем как в Whisper: нормализация по каждому примеру отдельно
        # Это предотвращает проблемы с NaN когда разные примеры имеют разные масштабы
        # Нормализуем по последним двум осям (n_mels, time_frames) для каждого примера
        # Используем reshape вместо view для совместимости с не-contiguous тензорами
        mel_spec_flat = mel_spec.reshape(batch_size, -1)  # (batch_size, n_mels * time_frames)
        mel_spec_mean = mel_spec_flat.mean(dim=1, keepdim=True)  # (batch_size, 1)
        mel_spec_std = mel_spec_flat.std(dim=1, keepdim=True) + 1e-5  # Увеличиваем epsilon для стабильности
        
        # Применяем нормализацию
        mel_spec = (mel_spec - mel_spec_mean.reshape(batch_size, 1, 1)) / mel_spec_std.reshape(batch_size, 1, 1)
        
        # Clipping для предотвращения экстремальных значений
        mel_spec = torch.clamp(mel_spec, min=-10.0, max=10.0)
        
        # Whisper энкодер ожидает input_features в формате (batch_size, n_mels, time_frames)
        # где n_mels=80 и time_frames должно быть 3000
        # Добавляем паддинг до нужной длины ПОСЛЕ нормализации
        target_time_frames = 3000
        current_time_frames = mel_spec.shape[2]  # последняя ось - time_frames
        
        if current_time_frames < target_time_frames:
            # Паддинг справа по временной оси (последняя ось)
            # Используем нули для паддинга (после нормализации это нормально)
            pad_length = target_time_frames - current_time_frames
            mel_spec = F.pad(mel_spec, (0, pad_length), mode='constant', value=0.0)
        elif current_time_frames > target_time_frames:
            # Обрезаем до 3000
            mel_spec = mel_spec[:, :, :target_time_frames]
        
        # Финальная проверка на NaN и Inf
        if torch.isnan(mel_spec).any() or torch.isinf(mel_spec).any():
            # Если есть NaN/Inf, заменяем на нули
            mel_spec = torch.nan_to_num(mel_spec, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Применяем Whisper энкодер
        # Формат: (batch_size, n_mels, time_frames) = (batch_size, 80, 3000)
        if self.freeze_encoder:
            with torch.inference_mode():
                encoder_outputs = self.whisper_encoder(input_features=mel_spec)
        elif not self.training:
            with torch.no_grad():
                encoder_outputs = self.whisper_encoder(input_features=mel_spec)
        else:
            encoder_outputs = self.whisper_encoder(input_features=mel_spec)
        
        # Получаем скрытые состояния энкодера
        # encoder_outputs.last_hidden_state shape: (batch_size, time_frames, d_model)
        encoder_features = encoder_outputs.last_hidden_state
        
        # Глобальное усреднение по временной оси
        audio_features = encoder_features.mean(dim=1)  # (batch_size, d_model)
        
        # Проверка на NaN после энкодера
        if torch.isnan(audio_features).any() or torch.isinf(audio_features).any():
            audio_features = torch.nan_to_num(audio_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Объединяем с метаданными
        if self.use_metadata_features and features is not None:
            combined_features = torch.cat([audio_features, features], dim=1)
        else:
            combined_features = audio_features
        
        # Проекция входных признаков
        x = self.input_projection(combined_features)
        
        # Применяем полносвязные слои с residual connections
        for layer in self.fc_layers:
            if self.use_residual:
                x = x + layer(x)  # Residual connection
            else:
                x = layer(x)
        
        # Проверка на NaN после fc_layers
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Предсказание STOI
        stoi_pred = self.output_layer(x)
        
        # Финальная проверка на NaN
        if torch.isnan(stoi_pred).any() or torch.isinf(stoi_pred).any():
            stoi_pred = torch.nan_to_num(stoi_pred, nan=0.5, posinf=1.0, neginf=0.0)
        
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
