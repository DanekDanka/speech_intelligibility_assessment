import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import wavfile

# Data paths
data_dir = "/home/danya/develop/datasets/CMU-MOSEI/Audio"
csv_path = os.path.join(data_dir, "stoi_results.csv")
vad_dir = os.path.join(data_dir, "vad")
output_dir = "images"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load data
df = pd.read_csv(csv_path)

# 1. STOI score distribution plot
plt.figure(figsize=(12, 6))
sns.histplot(df['stoi_score'], bins=30, kde=True)
plt.title('STOI Score Distribution')
plt.xlabel('STOI score')
plt.ylabel('Number of files')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'stoi_distribution.png'))
plt.close()

# 2. Total audio data amount (in minutes)
def get_audio_duration(wav_path):
    try:
        sample_rate, data = wavfile.read(wav_path)
        return len(data) / sample_rate / 60  # in minutes
    except:
        return 0

# Get durations for all files
df['duration_min'] = df['filename'].apply(
    lambda x: get_audio_duration(os.path.join(vad_dir, x))
)

total_minutes = df['duration_min'].sum()
total_files = len(df)

# Total audio data plot
plt.figure(figsize=(8, 6))
plt.bar(['Total duration'], [total_minutes], color='skyblue')
plt.ylabel('Minutes')
plt.title(f'Total Audio Data\n{total_minutes:.2f} minutes, {total_files} files')
plt.grid(True, axis='y')
plt.savefig(os.path.join(output_dir, 'total_audio_duration.png'))
plt.close()

# 3. Audio file duration distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['duration_min'], bins=30, kde=True)
plt.title('Audio File Duration Distribution')
plt.xlabel('Duration (minutes)')
plt.ylabel('Number of files')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'duration_distribution.png'))
plt.close()

# 4. STOI vs duration relationship
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='duration_min', y='stoi_score', alpha=0.6)
plt.title('STOI Score vs File Duration')
plt.xlabel('Duration (minutes)')
plt.ylabel('STOI score')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'stoi_vs_duration.png'))
plt.close()

# 5. STOI score boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, y='stoi_score')
plt.title('STOI Score Boxplot')
plt.ylabel('STOI score')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'stoi_boxplot.png'))
plt.close()

# 6. Top 10 files with highest and lowest STOI scores
top10_high = df.nlargest(10, 'stoi_score')
top10_low = df.nsmallest(10, 'stoi_score')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
sns.barplot(data=top10_high, x='filename', y='stoi_score', ax=ax1, palette='viridis')
ax1.set_title('Top 10 Files with Highest STOI Scores')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.grid(True)

sns.barplot(data=top10_low, x='filename', y='stoi_score', ax=ax2, palette='magma')
ax2.set_title('Top 10 Files with Lowest STOI Scores')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'top10_stoi.png'))
plt.close()

print(f"Plots successfully saved to {output_dir}")
print(f"Total audio data: {total_minutes:.2f} minutes")
print(f"Number of files: {total_files}")
