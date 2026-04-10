"""
Зашумление и реверберация для VoxCeleb по той же схеме, что CMU-MOSEI
(add_noise.py / add_reverbiration.py / add_noise_reverbiration.py), но без VAD:
SNR считается по мощности всего клипа.

Вход: /home/danya/datasets/voxceleb
  - voxceleb1_dev/wav/.../*.wav
  - voxceleb1_test/wav/.../*.wav
  - voxceleb2_test/aac/.../*.m4a

Выход (по умолчанию): /home/danya/datasets/voxceleb_augmented
  - train/{noise,reverb,noise_reverb}/...  — всё, что не test
  - test/{noise,reverb,noise_reverb}/...  — voxceleb1_test и voxceleb2_test

Имена файлов совместимы с parse_* из view_dataset / plot_stoi_comparison.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
import warnings

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from noise import (
    add_white_noise_with_target_snr,
    calculate_power,
    calculate_snr_from_powers,
    get_snr_bin,
    get_target_snr_for_uniform_distribution,
)
from reverbiration import (
    add_reverberation,
    get_rt60_bin,
    get_target_rt60_for_uniform_distribution,
)

INPUT_ROOT = Path("/home/danya/datasets/voxceleb")
OUTPUT_ROOT = Path("/home/danya/datasets/voxceleb_augmented")

AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp4", ".flac"}

SNR_MIN = -10.0
SNR_MAX = 20.0
SNR_BINS = 30

RT60_MIN = 0.1
RT60_MAX = 2.0
RT60_BINS = 30

WET_LEVEL_MIN = 0.1
WET_LEVEL_MAX = 0.8


def is_test_split(path: Path) -> bool:
    """По любому префиксу пути: voxceleb1_test / voxceleb2_test — отдельный test."""
    return any(p in ("voxceleb1_test", "voxceleb2_test") for p in path.resolve().parts)


def iter_audio_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            yield p


def _ffprobe_sample_rate(path: str) -> int:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    if not out:
        raise RuntimeError("ffprobe: no audio stream")
    return int(float(out.split()[0]))


def _load_via_ffmpeg(path: str, target_sr: int | None) -> tuple[np.ndarray, float]:
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        path,
        "-ac",
        "1",
        "-f",
        "f32le",
    ]
    if target_sr is not None:
        cmd.extend(["-ar", str(int(target_sr))])
    cmd.append("pipe:1")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    y = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if target_sr is not None:
        sr = float(target_sr)
    else:
        sr = float(_ffprobe_sample_rate(path))
    return y, sr


def load_audio(path: str | Path, target_sr: int | None) -> tuple[np.ndarray, float]:
    """
    WAV/FLAC — через soundfile (без попытки PySoundFile на m4a).
    m4a/mp4 — через ffmpeg (без audioread и ворнингов librosa).
    Иначе — librosa с подавлением известных предупреждений.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext in (".wav", ".flac"):
        y, sr = sf.read(str(path), dtype="float32", always_2d=False)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        sr = float(sr)
        if target_sr is not None and int(sr) != int(target_sr):
            y = librosa.resample(y, orig_sr=sr, target_sr=float(target_sr)).astype(np.float32)
            sr = float(target_sr)
        return y, sr

    if ext in (".m4a", ".mp4"):
        try:
            y, sr = _load_via_ffmpeg(str(path), target_sr)
        except (FileNotFoundError, subprocess.CalledProcessError, RuntimeError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                warnings.simplefilter("ignore", FutureWarning)
                y, sr = librosa.load(str(path), sr=target_sr, mono=True)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim > 1:
            y = y.flatten()
        return y, float(sr)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", FutureWarning)
        y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim > 1:
        y = y.flatten()
    return y, float(sr)


def full_signal_mask(n: int) -> np.ndarray:
    return np.ones(n, dtype=bool)


def fmt_param(x: float) -> str:
    return f"{x:.2f}".replace(".", "_")


def dist_path(split_dir: Path, sub: str, name: str) -> Path:
    return split_dir / sub / name


def load_json_dict(path: Path) -> defaultdict:
    if path.exists():
        with open(path, "r") as f:
            d = json.load(f)
        out = defaultdict(int)
        out.update(d)
        return out
    return defaultdict(int)


def save_json_dict(path: Path, d: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(dict(d), f, indent=2)


def process_one_file(
    in_file: Path,
    input_root: Path,
    output_root: Path,
    target_sr: int | None,
    snr_dist_train: defaultdict,
    snr_dist_test: defaultdict,
    rt60_dist_train: defaultdict,
    rt60_dist_test: defaultdict,
    combo_dist_train: defaultdict,
    combo_dist_test: defaultdict,
    do_noise: bool,
    do_reverb: bool,
    do_combo: bool,
) -> bool:
    try:
        y, sr = load_audio(str(in_file), target_sr)
        if len(y) == 0 or sr <= 0:
            return False

        test = is_test_split(in_file)
        split_name = "test" if test else "train"
        snr_d = snr_dist_test if test else snr_dist_train
        rt60_d = rt60_dist_test if test else rt60_dist_train
        combo_d = combo_dist_test if test else combo_dist_train

        rel = in_file.relative_to(input_root)
        out_stem = rel.stem + ".wav"

        base_out = output_root / split_name
        speech_mask = full_signal_mask(len(y))

        if do_noise:
            target_snr, _ = get_target_snr_for_uniform_distribution(
                snr_d, SNR_MIN, SNR_MAX, SNR_BINS
            )
            noisy = add_white_noise_with_target_snr(y, sr, target_snr, speech_mask)
            noise_sig = noisy - y
            sp_pow = calculate_power(y[speech_mask])
            actual_snr = calculate_snr_from_powers(sp_pow, calculate_power(noise_sig))
            bin_key = str(get_snr_bin(actual_snr, SNR_MIN, SNR_MAX, SNR_BINS))
            snr_d[bin_key] = snr_d.get(bin_key, 0) + 1
            name = f"snr={fmt_param(actual_snr)}__name={out_stem}"
            outp = base_out / "noise" / rel.parent / name
            outp.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(outp), noisy, int(sr))

        if do_reverb:
            target_rt60, _ = get_target_rt60_for_uniform_distribution(
                rt60_d, RT60_MIN, RT60_MAX, RT60_BINS
            )
            wet = float(np.random.uniform(WET_LEVEL_MIN, WET_LEVEL_MAX))
            rev = add_reverberation(y, sr, rt60=target_rt60, wet_level=wet)
            bin_key = str(get_rt60_bin(target_rt60, RT60_MIN, RT60_MAX, RT60_BINS))
            rt60_d[bin_key] = rt60_d.get(bin_key, 0) + 1
            name = f"rt60={fmt_param(target_rt60)}__wet={fmt_param(wet)}__name={out_stem}"
            outp = base_out / "reverb" / rel.parent / name
            outp.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(outp), rev, int(sr))

        if do_combo:
            snr_sub = {}
            rt60_sub = {}
            for key, value in combo_d.items():
                if key.startswith("snr_"):
                    snr_sub[key.replace("snr_", "")] = value
                elif key.startswith("rt60_"):
                    rt60_sub[key.replace("rt60_", "")] = value
            target_snr, _ = get_target_snr_for_uniform_distribution(
                snr_sub, SNR_MIN, SNR_MAX, SNR_BINS
            )
            target_rt60, _ = get_target_rt60_for_uniform_distribution(
                rt60_sub, RT60_MIN, RT60_MAX, RT60_BINS
            )
            wet = float(np.random.uniform(WET_LEVEL_MIN, WET_LEVEL_MAX))
            noisy = add_white_noise_with_target_snr(y, sr, target_snr, speech_mask)
            proc = add_reverberation(noisy, sr, rt60=target_rt60, wet_level=wet)
            actual_snr_bin = get_snr_bin(target_snr, SNR_MIN, SNR_MAX, SNR_BINS)
            actual_rt60_bin = get_rt60_bin(target_rt60, RT60_MIN, RT60_MAX, RT60_BINS)
            combo_d[f"snr_{actual_snr_bin}"] = combo_d.get(f"snr_{actual_snr_bin}", 0) + 1
            combo_d[f"rt60_{actual_rt60_bin}"] = combo_d.get(f"rt60_{actual_rt60_bin}", 0) + 1
            name = (
                f"snr={fmt_param(target_snr)}__rt60={fmt_param(target_rt60)}__"
                f"wet={fmt_param(wet)}__name={out_stem}"
            )
            outp = base_out / "noise_reverb" / rel.parent / name
            outp.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(outp), proc, int(sr))

        return True
    except Exception as e:
        print(f"Error {in_file}: {e}")
        return False


def run():
    ap = argparse.ArgumentParser(description="VoxCeleb: шум и реверб (без VAD).")
    ap.add_argument("--input", type=Path, default=INPUT_ROOT)
    ap.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    ap.add_argument(
        "--target-sr",
        type=int,
        default=16000,
        help="Ресемплинг (как WAV_16000 у MOSEI). 0 = оставить исходный SR.",
    )
    ap.add_argument("--noise-only", action="store_true")
    ap.add_argument("--reverb-only", action="store_true")
    ap.add_argument("--combo-only", action="store_true")
    args = ap.parse_args()

    input_root = args.input.resolve()
    output_root = args.output.resolve()
    target_sr = None if args.target_sr == 0 else args.target_sr

    do_noise = do_reverb = do_combo = True
    if args.noise_only or args.reverb_only or args.combo_only:
        do_noise = do_reverb = do_combo = False
        if args.noise_only:
            do_noise = True
        if args.reverb_only:
            do_reverb = True
        if args.combo_only:
            do_combo = True

    files = sorted(iter_audio_files(input_root))
    if not files:
        print(f"No audio files under {input_root}")
        return

    train_base = output_root / "train"
    test_base = output_root / "test"

    snr_dist_train = load_json_dict(dist_path(train_base, "noise", "snr_distribution.json"))
    snr_dist_test = load_json_dict(dist_path(test_base, "noise", "snr_distribution.json"))
    rt60_dist_train = load_json_dict(dist_path(train_base, "reverb", "rt60_distribution.json"))
    rt60_dist_test = load_json_dict(dist_path(test_base, "reverb", "rt60_distribution.json"))
    combo_dist_train = load_json_dict(
        dist_path(train_base, "noise_reverb", "distribution.json")
    )
    combo_dist_test = load_json_dict(dist_path(test_base, "noise_reverb", "distribution.json"))

    print(f"Input:  {input_root}")
    print(f"Output: {output_root} (train/ + test/)")
    print(f"Files:  {len(files)}")
    print(f"Target SR: {target_sr or 'native'}")
    print(f"Modes: noise={do_noise} reverb={do_reverb} combo={do_combo}")

    ok = fail = 0
    for fpath in tqdm(files, desc="VoxCeleb augment"):
        if process_one_file(
            fpath,
            input_root,
            output_root,
            target_sr,
            snr_dist_train,
            snr_dist_test,
            rt60_dist_train,
            rt60_dist_test,
            combo_dist_train,
            combo_dist_test,
            do_noise,
            do_reverb,
            do_combo,
        ):
            ok += 1
            if ok % 100 == 0:
                if do_noise:
                    save_json_dict(dist_path(train_base, "noise", "snr_distribution.json"), snr_dist_train)
                    save_json_dict(dist_path(test_base, "noise", "snr_distribution.json"), snr_dist_test)
                if do_reverb:
                    save_json_dict(
                        dist_path(train_base, "reverb", "rt60_distribution.json"), rt60_dist_train
                    )
                    save_json_dict(
                        dist_path(test_base, "reverb", "rt60_distribution.json"), rt60_dist_test
                    )
                if do_combo:
                    save_json_dict(
                        dist_path(train_base, "noise_reverb", "distribution.json"), combo_dist_train
                    )
                    save_json_dict(
                        dist_path(test_base, "noise_reverb", "distribution.json"), combo_dist_test
                    )
        else:
            fail += 1

    if do_noise:
        save_json_dict(dist_path(train_base, "noise", "snr_distribution.json"), snr_dist_train)
        save_json_dict(dist_path(test_base, "noise", "snr_distribution.json"), snr_dist_test)
    if do_reverb:
        save_json_dict(dist_path(train_base, "reverb", "rt60_distribution.json"), rt60_dist_train)
        save_json_dict(dist_path(test_base, "reverb", "rt60_distribution.json"), rt60_dist_test)
    if do_combo:
        save_json_dict(dist_path(train_base, "noise_reverb", "distribution.json"), combo_dist_train)
        save_json_dict(dist_path(test_base, "noise_reverb", "distribution.json"), combo_dist_test)

    print(f"Done. ok={ok} failed={fail}")


if __name__ == "__main__":
    run()
