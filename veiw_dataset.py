import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile

# Пути к данным
data_dir = "/home/danya/develop/datasets/CMU-MOSEI/Audio"
csv_path = os.path.join(data_dir, "stoi_results.csv")
vad_dir = os.path.join(data_dir, "vad")
output_dir = "images"

# Создаем папку для изображений, если ее нет
os.makedirs(output_dir, exist_ok=True)

# Загружаем данные
df = pd.read_csv(csv_path)

# 1. График распределения значений STOI
plt.figure(figsize=(12, 6))
sns.histplot(df['stoi_score'], bins=30, kde=True)
plt.title('Распределение значений STOI')
plt.xlabel('STOI score')
plt.ylabel('Количество файлов')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'stoi_distribution.png'))
plt.close()

# 2. Общее количество аудиоданных (в минутах)
def get_audio_duration(wav_path):
    try:
        sample_rate, data = wavfile.read(wav_path)
        return len(data) / sample_rate / 60  # в минутах
    except:
        return 0

# Получаем длительности всех файлов
df['duration_min'] = df['filename'].apply(
    lambda x: get_audio_duration(os.path.join(vad_dir, x))
)

total_minutes = df['duration_min'].sum()
total_files = len(df)

# График общего количества аудиоданных
plt.figure(figsize=(8, 6))
plt.bar(['Total duration'], [total_minutes], color='skyblue')
plt.ylabel('Минуты')
plt.title(f'Общее количество аудиоданных\n{total_minutes:.2f} минут, {total_files} файлов')
plt.grid(True, axis='y')
plt.savefig(os.path.join(output_dir, 'total_audio_duration.png'))
plt.close()

# 3. Распределение длительности аудиофайлов
plt.figure(figsize=(12, 6))
sns.histplot(df['duration_min'], bins=30, kde=True)
plt.title('Распределение длительности аудиофайлов')
plt.xlabel('Длительность (минуты)')
plt.ylabel('Количество файлов')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'duration_distribution.png'))
plt.close()

# 4. Зависимость STOI от длительности файла
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='duration_min', y='stoi_score', alpha=0.6)
plt.title('Зависимость STOI от длительности файла')
plt.xlabel('Длительность (минуты)')
plt.ylabel('STOI score')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'stoi_vs_duration.png'))
plt.close()

# 5. Boxplot распределения STOI
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, y='stoi_score')
plt.title('Boxplot распределения STOI')
plt.ylabel('STOI score')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'stoi_boxplot.png'))
plt.close()

# 6. Топ-10 файлов с наивысшим и наинизшим STOI
top10_high = df.nlargest(10, 'stoi_score')
top10_low = df.nsmallest(10, 'stoi_score')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
sns.barplot(data=top10_high, x='filename', y='stoi_score', ax=ax1, palette='viridis')
ax1.set_title('Топ-10 файлов с наивысшим STOI')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.grid(True)

sns.barplot(data=top10_low, x='filename', y='stoi_score', ax=ax2, palette='magma')
ax2.set_title('Топ-10 файлов с наинизшим STOI')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top10_stoi.png'))
plt.close()

print(f"Графики успешно сохранены в {output_dir}")
print(f"Общее количество аудиоданных: {total_minutes:.2f} минут")
print(f"Количество файлов: {total_files}")
