import os
import numpy as np
import soundfile as sf
from pystoi import stoi
import csv
from datetime import datetime
from tqdm import tqdm

def calculate_stoi(clean_path, noisy_path):
    clean, sr_clean = sf.read(clean_path)
    noisy, sr_noisy = sf.read(noisy_path)
    
    if sr_clean != sr_noisy:
        raise ValueError("Sample rates don't match!")
    
    min_len = min(len(clean), len(noisy))
    clean = clean[:min_len]
    noisy = noisy[:min_len]
    
    return stoi(clean, noisy, sr_clean, extended=False)

def evaluate_stoi_to_csv(clean_dir, noisy_dir, output_csv):
    clean_files = sorted([f for f in os.listdir(clean_dir) if f.lower().endswith(('.wav', '.mp3', '.flac'))])
    noisy_files = sorted([f for f in os.listdir(noisy_dir) if f.lower().endswith(('.wav', '.mp3', '.flac'))])
    
    if len(clean_files) != len(noisy_files):
        print("Warning: Different number of files in directories!")
    
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'stoi_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        stoi_scores = []
        errors = 0
        
        for clean_file, noisy_file in tqdm(zip(clean_files, noisy_files), 
                                         total=len(clean_files),
                                         desc="Processing files"):
            clean_path = os.path.join(clean_dir, clean_file)
            noisy_path = os.path.join(noisy_dir, noisy_file)
            
            try:
                score = calculate_stoi(clean_path, noisy_path)
                stoi_scores.append(score)
                
                writer.writerow({
                    'filename': clean_file,
                    'stoi_score': f"{score:.4f}"
                })
                
            except Exception as e:
                errors += 1
                writer.writerow({
                    'filename': clean_file,
                    'stoi_score': 'error'
                })
        
        if stoi_scores:
            mean_stoi = np.mean(stoi_scores)
            print(f"\nProcessing complete:")
            print(f"Total files: {len(clean_files)}")
            print(f"Successfully processed: {len(stoi_scores)}")
            print(f"Errors: {errors}")
            print(f"Average STOI: {mean_stoi:.4f}")
            print(f"Results saved to: {output_csv}")

if __name__ == "__main__":
    clean_dir = "/home/danya/develop/datasets/CMU-MOSEI/Audio/vad/"
#     noisy_dir = "/home/danya/develop/datasets/CMU-MOSEI/Audio/vad_noise/"
    noisy_dir = "/home/danya/develop/datasets/CMU-MOSEI/Audio/vad_reverberation/"
    output_csv = "/home/danya/develop/datasets/CMU-MOSEI/Audio/stoi_results.csv"
    
    evaluate_stoi_to_csv(clean_dir, noisy_dir, output_csv)
