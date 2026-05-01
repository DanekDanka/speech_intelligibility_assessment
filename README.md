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
pip install -r demonstration_program/requirements.txt
cd demonstration_program
./run.sh hf://DanekDanka/NI-STOI/best.pt
```
