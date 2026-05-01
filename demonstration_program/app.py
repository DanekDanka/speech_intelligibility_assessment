"""
Демонстрационная программа для показа работы модели предсказания STOI (STOI-Net).
Использует Gradio для создания веб-интерфейса.

По умолчанию: конфиг ``src_STOI/configs/train_stoi_net.json`` и чекпоинт
``checkpoints_src_stoi_net/best.pt`` (как при обучении).

Переопределение:
  ``STOI_NET_CONFIG`` — путь к JSON конфигу обучения;
  ``MODEL_CHECKPOINT`` — путь к ``best.pt`` (или другому ckpt в том же формате).
"""
from __future__ import annotations

import os
import sys
import torch
import numpy as np
import librosa
from scipy.signal import convolve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Используем non-interactive backend
from pathlib import Path
import gradio as gr

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src_STOI.config_utils import load_merged_config
from src_STOI.model import build_model

_DEFAULT_CONFIG_PATH = _REPO_ROOT / "src_STOI" / "configs" / "train_stoi_net.json"
_DEFAULT_CHECKPOINT_PATH = _REPO_ROOT / "checkpoints_src_stoi_net" / "best.pt"


def _resolve_under_repo(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    return path


_cfg_path = os.environ.get("STOI_NET_CONFIG")
CONFIG_PATH = _resolve_under_repo(Path(_cfg_path)) if _cfg_path else _DEFAULT_CONFIG_PATH
_ckpt_path = os.environ.get("MODEL_CHECKPOINT")
CHECKPOINT_PATH = _resolve_under_repo(Path(_ckpt_path)) if _ckpt_path else _DEFAULT_CHECKPOINT_PATH

_TRAIN_CFG = load_merged_config([str(CONFIG_PATH)])
_data_cfg = _TRAIN_CFG["data"]

# Параметры как в обучении (train_stoi_net.json)
SAMPLE_RATE = int(_data_cfg["sample_rate"])
CHUNK_DURATION = float(_data_cfg["chunk_duration_sec"])
CHUNK_STEP = 1.0  # Шаг в 1 секунду (только для демо)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Глобальная переменная для хранения модели (загружается один раз)
_loaded_model = None


def load_model(checkpoint_path: str | Path | None = None):
    """Загружает STOI-Net из чекпоинта (формат ``train.py`` / ``test_model.py``)."""
    ckpt_path = Path(checkpoint_path) if checkpoint_path else CHECKPOINT_PATH
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Чекпоинт не найден: {ckpt_path}")

    print(f"Загрузка модели из {ckpt_path}...")
    try:
        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)

    if "model" not in checkpoint or "state_dict" not in checkpoint:
        raise KeyError(
            f"Ожидались ключи 'model' и 'state_dict' в чекпоинте (как у checkpoints_src_stoi_net). "
            f"Получены: {list(checkpoint.keys())}"
        )

    m_cfg = checkpoint["model"]
    model = build_model(m_cfg["name"], m_cfg.get("kwargs") or {})
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(DEVICE)
    model.eval()
    print(
        f"Модель «{m_cfg['name']}» загружена. Параметров: {sum(p.numel() for p in model.parameters()):,}"
    )
    return model


def add_reverb_simple(
    y: np.ndarray,
    sr: float,
    rt60: float,
    wet_level: float,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Простая синтетическая реверберация (случайная ИХ + экспоненциальное затухание),
    по идее как в ``scripts/reverbiration.py``.
    """
    rng = rng or np.random.default_rng()
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    rt60 = float(np.clip(rt60, 0.02, 3.0))
    wet_level = float(np.clip(wet_level, 0.0, 1.0))
    sr = float(sr)
    length = max(int(rt60 * sr), 64)
    ir = rng.standard_normal(length).astype(np.float32)
    decay = np.exp(-np.linspace(0.0, 10.0, length, dtype=np.float32))
    ir *= decay
    m = float(np.max(np.abs(ir)))
    if m > 0:
        ir /= m
    wet = convolve(y, ir, mode="same").astype(np.float32)
    dry_level = 1.0 - wet_level
    out = (dry_level * y + wet_level * wet).astype(np.float32)
    peak = float(np.max(np.abs(out)))
    if peak > 0:
        out /= peak
    return np.clip(out, -1.0, 1.0)


def add_awgn_snr(y: np.ndarray, snr_db: float, *, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Белый гауссов шум; SNR в дБ относительно средней мощности сигнала (по всему чанку).
    """
    rng = rng or np.random.default_rng()
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    p_sig = float(np.mean(y.astype(np.float64) ** 2))
    if p_sig < 1e-14:
        return y.copy()
    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    p_noise = p_sig / snr_lin
    noise = rng.standard_normal(len(y)).astype(np.float32) * float(np.sqrt(p_noise))
    out = np.clip(y + noise, -1.0, 1.0)
    peak = float(np.max(np.abs(out)))
    if peak > 1.0:
        out = (out / peak).astype(np.float32)
    return out


def split_audio_into_chunks(audio, sample_rate, chunk_duration=5.0, step=1.0):
    """
    Разделяет аудио на чанки заданной длительности с заданным шагом.
    
    Args:
        audio: numpy array с аудио данными
        sample_rate: частота дискретизации
        chunk_duration: длительность чанка в секундах
        step: шаг между чанками в секундах
    
    Returns:
        chunks: список чанков (каждый как numpy array)
        chunk_times: список временных меток начала каждого чанка (в секундах)
    """
    chunk_samples = int(chunk_duration * sample_rate)
    step_samples = int(step * sample_rate)
    
    chunks = []
    chunk_times = []
    
    start_idx = 0
    while start_idx + chunk_samples <= len(audio):
        chunk = audio[start_idx:start_idx + chunk_samples]
        chunks.append(chunk)
        chunk_times.append(start_idx / sample_rate)
        start_idx += step_samples
    
    return chunks, chunk_times


def predict_stoi(model, audio_chunk, sample_rate=SAMPLE_RATE):
    """
    Предсказывает STOI для одного чанка аудио.
    
    Args:
        model: обученная модель
        audio_chunk: numpy array с аудио данными
        sample_rate: частота дискретизации
    
    Returns:
        stoi_pred: предсказанное значение STOI
    """
    # Нормализуем аудио
    if len(audio_chunk) == 0:
        return 0.0
    
    # Ресемплинг до нужной частоты, если необходимо
    if sample_rate != SAMPLE_RATE:
        audio_chunk = librosa.resample(audio_chunk, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    
    # Конвертируем в tensor
    audio_tensor = torch.FloatTensor(audio_chunk).to(DEVICE)
    
    # Добавляем batch dimension
    audio_tensor = audio_tensor.unsqueeze(0)  # (1, seq_len)
    
    # Предсказание (STOI-Net: выход [B, 1])
    with torch.no_grad():
        stoi_pred = model(audio_tensor).view(-1)
        stoi_pred = float(stoi_pred[0].cpu())
    return stoi_pred


def _load_audio_mono(audio_file) -> tuple[np.ndarray, int]:
    if isinstance(audio_file, tuple):
        sample_rate, audio_data = audio_file
        if audio_data.ndim > 1:
            audio = audio_data[:, 0].astype(np.float32)
        else:
            audio = audio_data.astype(np.float32)
    else:
        audio, sample_rate = librosa.load(audio_file, sr=None, mono=True)
    return audio, int(sample_rate)


def process_audio(
    audio_file,
    use_reverb: bool,
    rt60: float,
    wet_level: float,
    use_noise: bool,
    snr_db: float,
):
    """
    Обрабатывает аудио: опционально реверберация и шум, затем предсказание STOI.

    Порядок эффектов: нормализация → реверберация → шум (как типичная цепочка деградации).
    """
    if audio_file is None:
        return None, "Загрузите аудио файл", None, None

    try:
        audio, sample_rate = _load_audio_mono(audio_file)

        duration = len(audio) / sample_rate
        if duration < CHUNK_DURATION:
            return None, (
                f"Аудио слишком короткое ({duration:.2f} с). Нужно минимум {CHUNK_DURATION} с."
            ), None, None

        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        if sample_rate != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE).astype(np.float32)
            sample_rate = SAMPLE_RATE

        fx_rng = np.random.default_rng()
        effects_lines: list[str] = []
        if use_reverb:
            audio = add_reverb_simple(audio, float(sample_rate), rt60, wet_level, rng=fx_rng)
            effects_lines.append(
                f"- **Реверберация:** RT60 = {rt60:.2f} с, уровень смеси (мокрый) = {wet_level:.2f}"
            )
        if use_noise:
            audio = add_awgn_snr(audio, float(snr_db), rng=fx_rng)
            effects_lines.append(f"- **Шум:** SNR ≈ {snr_db:.1f} дБ (белый шум к мощности сигнала)")
        if not effects_lines:
            effects_lines.append("- **Эффекты:** выключены (исходная запись)")

        # Разделяем на чанки
        chunks, chunk_times = split_audio_into_chunks(
            audio, sample_rate, CHUNK_DURATION, CHUNK_STEP
        )
        
        if len(chunks) == 0:
            return None, f"Аудио слишком короткое (нужно минимум {CHUNK_DURATION} с)", None, None
        
        # Загружаем модель (делаем это один раз при первом вызове)
        global _loaded_model
        if _loaded_model is None:
            _loaded_model = load_model(CHECKPOINT_PATH)
        
        # Предсказываем STOI для каждого чанка
        stoi_predictions = []
        for chunk in chunks:
            stoi = predict_stoi(_loaded_model, chunk, sample_rate)
            stoi_predictions.append(stoi)
        
        stoi_predictions = np.array(stoi_predictions)
        chunk_times = np.array(chunk_times)
        
        # Вычисляем среднее значение
        mean_stoi = np.mean(stoi_predictions)
        
        # Создаем график STOI
        fig_stoi, ax_stoi = plt.subplots(figsize=(10, 6))
        ax_stoi.plot(chunk_times, stoi_predictions, 'b-o', linewidth=2, markersize=8)
        ax_stoi.axhline(y=mean_stoi, color='r', linestyle='--', linewidth=2, label=f'Среднее: {mean_stoi:.3f}')
        ax_stoi.set_xlabel('Время (секунды)', fontsize=12)
        ax_stoi.set_ylabel('STOI', fontsize=12)
        title_stoi = "Предсказанные значения STOI по времени"
        if use_reverb or use_noise:
            title_stoi += " (после эффектов)"
        ax_stoi.set_title(title_stoi, fontsize=14, fontweight='bold')
        ax_stoi.grid(True, alpha=0.3)
        ax_stoi.legend(fontsize=11)
        ax_stoi.set_ylim([0, 1])
        plt.tight_layout()
        
        # Создаем мел-спектр с наложенными значениями STOI
        # Используем весь аудио сигнал для мел-спектра
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sample_rate, n_mels=128, hop_length=512
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        fig_mel, ax_mel = plt.subplots(figsize=(12, 6))
        
        # Отображаем мел-спектр
        times_mel = librosa.frames_to_time(np.arange(mel_spec_db.shape[1]), sr=sample_rate, hop_length=512)
        im = ax_mel.imshow(mel_spec_db, aspect='auto', origin='lower', 
                          extent=[times_mel[0], times_mel[-1], 0, 128],
                          cmap='viridis', interpolation='bilinear')
        
        # Накладываем значения STOI
        # Для каждого чанка рисуем прямоугольник с цветом, соответствующим значению STOI
        for i, (chunk_time, stoi_val) in enumerate(zip(chunk_times, stoi_predictions)):
            chunk_end = chunk_time + CHUNK_DURATION
            # Используем цветовую карту для отображения STOI (зеленый = высокий, красный = низкий)
            # Нормализуем STOI для цветовой карты (STOI в диапазоне 0-1)
            color = plt.cm.RdYlGn(stoi_val)  # Red-Yellow-Green colormap
            # Рисуем полупрозрачный прямоугольник
            rect = plt.Rectangle((chunk_time, 0), CHUNK_DURATION, 128, 
                               facecolor=color, alpha=0.4, edgecolor='white', linewidth=1.5)
            ax_mel.add_patch(rect)
            # Добавляем текст с значением STOI (только если чанк достаточно большой)
            if CHUNK_DURATION >= 2.0:
                text_color = 'white' if stoi_val < 0.5 else 'black'
                ax_mel.text(chunk_time + CHUNK_DURATION / 2, 64, f'{stoi_val:.2f}',
                           ha='center', va='center', fontsize=9, fontweight='bold',
                           color=text_color, bbox=dict(boxstyle='round,pad=0.3', 
                           facecolor='white', alpha=0.7, edgecolor='black', linewidth=0.5))
        
        # Добавляем цветовую шкалу для STOI
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar2 = plt.colorbar(sm, ax=ax_mel, label='STOI', location='right', pad=0.02)
        cbar2.set_label('STOI (наложено на спектр)', rotation=270, labelpad=20)
        
        ax_mel.set_xlabel('Время (секунды)', fontsize=12)
        ax_mel.set_ylabel('Частота (мел-бин)', fontsize=12)
        title_mel = "Мел-спектр с наложенными значениями STOI"
        if use_reverb or use_noise:
            title_mel += " (обработанный сигнал)"
        ax_mel.set_title(title_mel, fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax_mel, label='Амплитуда (dB)')
        plt.tight_layout()
        
        effects_block = "\n".join(effects_lines)
        result_text = f"""
**Обработка сигнала**

{effects_block}

**Результаты модели**

- **Количество чанков:** {len(chunks)}
- **Длительность каждого чанка:** {CHUNK_DURATION} с
- **Шаг между чанками:** {CHUNK_STEP} с
- **Среднее значение STOI:** {mean_stoi:.4f}
- **Минимальное значение STOI:** {np.min(stoi_predictions):.4f}
- **Максимальное значение STOI:** {np.max(stoi_predictions):.4f}
- **Стандартное отклонение:** {np.std(stoi_predictions):.4f}
        """

        processed_for_player = (sample_rate, audio.reshape(-1, 1).astype(np.float32))

        return fig_stoi, result_text, fig_mel, processed_for_player

    except Exception as e:
        import traceback
        error_msg = f"Ошибка при обработке аудио: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg, None, None


# Создаем интерфейс Gradio
def create_interface():
    """Создает интерфейс Gradio"""
    
    with gr.Blocks(title="STOI Prediction Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"""
        # 🎤 Демонстрация модели предсказания STOI 🎤
        
        Эта программа оценивает разборчивость речи (STOI) с помощью **STOI-Net**, обученной по конфигу
        ``{CONFIG_PATH.name}`` (частота {SAMPLE_RATE} Гц, длина чанка {CHUNK_DURATION} с).
        
        **Инструкция:**
        1. Запишите или загрузите аудио (минимум **{CHUNK_DURATION}** с).
        2. При необходимости включите **реверберацию** и/или **шум** и настройте параметры.
        3. Нажмите **«Обработать аудио»** — к сигналу применятся эффекты, затем **STOI-Net** оценит каждый чанк ({CHUNK_DURATION} с, шаг {CHUNK_STEP} с).
        4. Ниже — графики и **прослушивание обработанного** сигнала.
        """)
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    label="Исходное аудио (запись или файл)",
                    type="numpy",
                    sources=["microphone", "upload"],
                )
                with gr.Accordion("Реверберация и шум", open=True):
                    use_reverb = gr.Checkbox(label="Включить реверберацию", value=False)
                    rt60 = gr.Slider(
                        label="RT60 (с) — длительность «хвоста»",
                        minimum=0.1,
                        maximum=1.5,
                        value=0.5,
                        step=0.05,
                    )
                    wet_level = gr.Slider(
                        label="Доля «мокрого» сигнала (0 = только сухой, 1 = только реверб)",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.35,
                        step=0.05,
                    )
                    use_noise = gr.Checkbox(label="Включить белый шум (AWGN)", value=False)
                    snr_db = gr.Slider(
                        label="SNR (дБ) — отношение сигнал/шум по мощности",
                        minimum=-5.0,
                        maximum=40.0,
                        value=15.0,
                        step=0.5,
                    )
                process_btn = gr.Button("Обработать аудио", variant="primary", size="lg")

            with gr.Column():
                results_text = gr.Markdown(label="Результаты")
                audio_processed = gr.Audio(
                    label="Обработанное аудио (после эффектов, то же что у модели)",
                    type="numpy",
                    interactive=False,
                )

        with gr.Row():
            stoi_plot = gr.Plot(label="График значений STOI")
            mel_spectrogram = gr.Plot(label="Мел-спектр с наложенными значениями STOI")

        process_inputs = [
            audio_input,
            use_reverb,
            rt60,
            wet_level,
            use_noise,
            snr_db,
        ]
        process_outputs = [stoi_plot, results_text, mel_spectrogram, audio_processed]

        process_btn.click(fn=process_audio, inputs=process_inputs, outputs=process_outputs)
        
        # gr.Markdown("""
        # ---
        # **Примечание:** Модель работает лучше всего с чистой речью без сильных шумов.
        # Рекомендуется записывать в тихой обстановке с хорошим микрофоном.
        # """)
    
    return demo


if __name__ == "__main__":
    print(f"Конфиг обучения: {CONFIG_PATH}")
    print(f"Чекпоинт:        {CHECKPOINT_PATH}")
    if not CHECKPOINT_PATH.is_file():
        print(f"⚠️  Внимание: чекпоинт не найден: {CHECKPOINT_PATH}")
        print("Укажите MODEL_CHECKPOINT или обучите модель (output_dir в train_stoi_net.json).")
        print("Пример: export MODEL_CHECKPOINT=checkpoints_src_stoi_net/best.pt")
    
    # Создаем и запускаем интерфейс
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",  # Доступно извне
        server_port=7860,
        share=False  # Установите True для создания публичной ссылки
    )
