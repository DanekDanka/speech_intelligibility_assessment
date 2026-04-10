# Speech intelligibility assessment

This repo trains neural models to estimate **speech intelligibility** from **degraded audio alone** (noise, reverberation, etc.) on datasets such as **CMU-MOSEI** and **VoxCeleb**.

The **`src_STOI`** track learns to predict **STOI-like scores** from short wav chunks. That is a **reference-free (non-intrusive) analogue of STOI**: classic STOI needs a **clean reference** utterance aligned with the processed signal; here the model is trained to output a similar score **using only the degraded waveform** at inference time—no clean reference is required.

**Layout:**

- `src/` — datasets, CNN / LSTM / Transformer / Whisper training code from the main line of work.
- `src_STOI/` — configs, `train.py`, STOI-related metrics, regression model training.
- `scripts/` — helpers (noise, reverb, dataset inspection).
- `vew_some_wav/` — ad hoc experiments on individual wav files.

**Quick start (STOI predictor):**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r src_STOI/requirements.txt
python src_STOI/train.py --config src_STOI/configs/train_stoi_net.json
```

Dataset paths live in the JSON configs. Checkpoints, TensorBoard logs, and STOI label caches are not meant to be committed (see `.gitignore`).
