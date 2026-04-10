from __future__ import annotations

import math
import multiprocessing as mp
import pickle
import re
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torchaudio
from torch.utils.data import Subset
from tqdm import tqdm

from ..preprocessing.base import StoiTargetComputer, WaveformPreprocessor
from .dataset_interface import StoiPredictionDataset
from .stoi_cache_worker import _pool_init, compute_stoi_group
from .stoi_rich_cache import (
    infer_noise_reverb_from_processed_path,
    load_rich_cache,
    merge_float_cache_for_keys,
    save_rich_cache,
)


def extract_original_filename(filename: str) -> str:
    match = re.search(r"__name=(.+)$", filename)
    if match:
        original_name = match.group(1)
        if not original_name.endswith(".wav"):
            return original_name + ".wav"
        return original_name
    if "__" in filename:
        return filename.split("__")[-1]
    return filename


def resolve_mirror_original_path(processed_wav: Path, mirror_root: Union[str, Path]) -> Optional[Path]:
    """
    Для дерева вида ``.../train|test/{noise,reverb,noise_reverb}/<как в voxceleb>/файл.wav``
    находит чистый клип под ``mirror_root`` с тем же относительным путём (без папки аугментации).
    Имя чистого файла — из ``__name=``; если ``*.wav`` нет, пробует тот же stem с ``.m4a`` / ``.flac``.
    """
    mirror_root = Path(mirror_root).resolve()
    parts = processed_wav.resolve().parts
    marker_idx: Optional[int] = None
    for i, p in enumerate(parts):
        if p in ("noise", "reverb", "noise_reverb"):
            marker_idx = i
            break
    if marker_idx is None or marker_idx + 1 >= len(parts):
        return None
    rel = Path(*parts[marker_idx + 1 :])
    clean_from_tag = extract_original_filename(processed_wav.name)
    stem = Path(clean_from_tag).stem
    parent = rel.parent
    seen: set[Path] = set()
    for name in (clean_from_tag, f"{stem}.wav", f"{stem}.m4a", f"{stem}.flac"):
        cand = mirror_root / parent / name
        if cand in seen:
            continue
        seen.add(cand)
        if cand.is_file():
            return cand
    return None


def _num_frames_without_decode(path: Path) -> int | None:
    """
    Длина в сэмплах без полной декодировки (torchaudio 2.9+ не даёт torchaudio.info).
    Сначала soundfile (заголовок), затем stdlib wave для простого PCM WAV.
    """
    p = str(path)
    try:
        import soundfile as sf

        return int(sf.info(p).frames)
    except Exception:
        pass
    try:
        with wave.open(p, "rb") as wf:
            return int(wf.getnframes())
    except Exception:
        pass
    return None


def _estimate_num_chunks(path: Path, chunk_samples: int) -> int:
    nf = _num_frames_without_decode(path)
    if nf is None:
        w, _sr = torchaudio.load(str(path))
        nf = int(w.shape[-1])
    if nf <= 0:
        return 1
    return max(1, math.ceil(nf / chunk_samples))


class PairWavStoiDataset(StoiPredictionDataset):
    """
    Pairs processed (degraded) WAV files with originals under ``original_dir``,
    same chunking as in ``src/dataset.py`` style manifests.

    Если задан ``mirror_original_root`` (VoxCeleb augmented), оригинал ищется через
    :func:`resolve_mirror_original_path`; иначе — плоский ``original_dir`` + rglob по имени.

    ``index_fraction`` < 1.0: после построения полного списка чанков оставляет случайную
    долю (детерминированно по ``index_sample_seed``), затем только для них считается STOI-кэш
    и обучение/сплиты.

    ``shuffle_chunks``: перемешать порядок чанков в индексе (после subsample), чтобы сплит
    и обход не шли строго по отсортированным путям файлов.
    """

    def __init__(
        self,
        audio_dirs: Sequence[Union[str, Path]],
        original_dir: Union[str, Path],
        waveform_preprocessor: WaveformPreprocessor,
        stoi_computer: StoiTargetComputer,
        *,
        mirror_original_root: Optional[Union[str, Path]] = None,
        single_chunk_per_file: bool = False,
        cache_stoi_path: Optional[Union[str, Path]] = None,
        max_items: Optional[int] = None,
        stoi_cache_num_workers: Optional[int] = None,
        index_fraction: float = 1.0,
        index_sample_seed: int = 0,
        shuffle_chunks: bool = True,
        rich_cache_path: Optional[Union[str, Path]] = None,
        import_legacy_cache_paths: Optional[Sequence[Union[str, Path]]] = None,
    ) -> None:
        self.original_dir = Path(original_dir)
        self.mirror_original_root = (
            Path(mirror_original_root).resolve() if mirror_original_root is not None else None
        )
        self.waveform_preprocessor = waveform_preprocessor
        self.stoi_computer = stoi_computer
        self.single_chunk_per_file = single_chunk_per_file
        self._stoi_cache_num_workers = stoi_cache_num_workers
        self.chunk_samples = getattr(waveform_preprocessor, "chunk_samples", None)
        if self.chunk_samples is None:
            raise TypeError("waveform_preprocessor must expose chunk_samples (use TorchaudioResampleMonoChunkPreprocessor).")

        audio_dirs = [Path(d) for d in audio_dirs]
        self.audio_files: List[Path] = []
        for d in audio_dirs:
            if not d.exists():
                continue
            self.audio_files.extend(sorted(d.rglob("*.wav")))
        if max_items is not None:
            self.audio_files = self.audio_files[: int(max_items)]

        self.pairs: List[Tuple[Path, Path]] = []
        for proc in self.audio_files:
            if self.mirror_original_root is not None:
                orig = resolve_mirror_original_path(proc, self.mirror_original_root)
                if orig is None:
                    continue
            else:
                orig_name = extract_original_filename(proc.name)
                orig = self.original_dir / orig_name
                if not orig.exists():
                    found = list(self.original_dir.rglob(orig_name))
                    if not found:
                        continue
                    orig = found[0]
            self.pairs.append((proc, orig))

        self.chunk_index: List[Tuple[Path, Path, int]] = []
        for proc, orig in self.pairs:
            n = 1 if self.single_chunk_per_file else _estimate_num_chunks(proc, self.chunk_samples)
            for c in range(n):
                self.chunk_index.append((proc, orig, c))

        frac = float(index_fraction)
        if frac < 1.0 - 1e-12:
            if frac <= 0 or frac > 1:
                raise ValueError("index_fraction must be in (0, 1]")
            n0 = len(self.chunk_index)
            if n0 > 0:
                k = max(1, int(n0 * frac))
                k = min(k, n0)
                g = torch.Generator().manual_seed(int(index_sample_seed))
                perm = torch.randperm(n0, generator=g).tolist()
                self.chunk_index = [self.chunk_index[i] for i in perm[:k]]

        if shuffle_chunks and len(self.chunk_index) > 1:
            g_sh = torch.Generator().manual_seed(int(index_sample_seed) + 18443)
            p_sh = torch.randperm(len(self.chunk_index), generator=g_sh).tolist()
            self.chunk_index = [self.chunk_index[i] for i in p_sh]

        self.cache_path = Path(cache_stoi_path) if cache_stoi_path else None
        self.rich_cache_path = Path(rich_cache_path) if rich_cache_path else None
        self._import_legacy_paths: List[Path] = (
            [Path(p) for p in import_legacy_cache_paths] if import_legacy_cache_paths else []
        )

        self._stoi_cache: Dict[str, float] = {}
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                raw = pickle.load(f)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    try:
                        self._stoi_cache[str(k)] = float(v)
                    except (TypeError, ValueError):
                        continue

        needed_keys = {self._cache_key(p, o, c) for p, o, c in self.chunk_index}
        for leg in self._import_legacy_paths:
            merge_float_cache_for_keys(leg, needed_keys, self._stoi_cache)

        self._rich_entries: Dict[str, Dict[str, Any]] = {}
        if self.rich_cache_path:
            self._rich_entries = load_rich_cache(self.rich_cache_path)
            for k in needed_keys:
                if k not in self._stoi_cache and k in self._rich_entries:
                    e = self._rich_entries[k]
                    sv = e.get("stoi")
                    if isinstance(sv, (int, float)):
                        self._stoi_cache[k] = float(sv)

        self._fill_stoi_cache()
        self._persist_rich_cache()

    def _effective_stoi_cache_workers(self) -> int:
        nw = self._stoi_cache_num_workers
        if nw is None or nw == 0:
            return max(1, min(16, (mp.cpu_count() or 8) - 1))
        return max(1, int(nw))

    def _cache_key(self, proc: Path, orig: Path, chunk: int) -> str:
        fill = "mp" if self._effective_stoi_cache_workers() > 1 else "seq"
        dev = getattr(self.stoi_computer, "stoi_compute_device", torch.device("cpu"))
        silrm = int(getattr(self.stoi_computer, "apply_silence_removal", True))
        return (
            f"{proc.resolve()}|{orig.resolve()}|c={chunk}|sr={self.stoi_computer.sample_rate}"
            f"|fill={fill}|dev={dev}|silrm={silrm}"
        )

    def _fill_stoi_cache(self) -> None:
        pending: DefaultDict[Tuple[Path, Path], List[Tuple[int, str]]] = defaultdict(list)
        for proc, orig, chunk in self.chunk_index:
            key = self._cache_key(proc, orig, chunk)
            if key not in self._stoi_cache:
                pending[(proc, orig)].append((chunk, key))

        total_missing = sum(len(v) for v in pending.values())
        if total_missing == 0:
            return

        nw = self._effective_stoi_cache_workers()

        repo_root = str(Path(__file__).resolve().parents[2])

        if nw <= 1:
            self._fill_stoi_cache_sequential_grouped(pending, total_missing)
        else:
            self._fill_stoi_cache_parallel_grouped(pending, total_missing, nw, repo_root)

        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._stoi_cache, f)

    def _fill_stoi_cache_sequential_grouped(
        self,
        pending: DefaultDict[Tuple[Path, Path], List[Tuple[int, str]]],
        total_missing: int,
    ) -> None:
        pre = self.waveform_preprocessor
        with tqdm(total=total_missing, desc="STOI метки (кэш)", unit="чанк", leave=True) as pbar:
            for (proc, orig), items in pending.items():
                ref_w = pre.load_mono_resampled(orig)
                deg_w = pre.load_mono_resampled(proc)
                for chunk, key in items:
                    ref = pre.chunk_waveform(ref_w, chunk)
                    deg = pre.chunk_waveform(deg_w, chunk)
                    val = self.stoi_computer.compute(ref, deg)
                    self._stoi_cache[key] = float(max(0.0, min(1.0, val)))
                    pbar.update(1)

    def _fill_stoi_cache_parallel_grouped(
        self,
        pending: DefaultDict[Tuple[Path, Path], List[Tuple[int, str]]],
        total_missing: int,
        nw: int,
        repo_root: str,
    ) -> None:
        extended = bool(getattr(self.stoi_computer, "extended", False))
        resample_mode = str(getattr(self.stoi_computer, "resample_mode", "torchaudio"))
        apply_silence_removal = bool(getattr(self.stoi_computer, "apply_silence_removal", True))
        sr = self.stoi_computer.sample_rate
        chunk_sec = float(self.waveform_preprocessor.chunk_duration_sec)

        payloads: List[Dict[str, Any]] = []
        for (proc, orig), items in pending.items():
            payloads.append(
                {
                    "_repo_root": repo_root,
                    "proc": str(proc.resolve()),
                    "orig": str(orig.resolve()),
                    "sample_rate": sr,
                    "chunk_duration_sec": chunk_sec,
                    "extended": extended,
                    "resample_mode": resample_mode,
                    "apply_silence_removal": apply_silence_removal,
                    "items": [{"chunk": c, "key": k} for c, k in items],
                }
            )

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=nw, initializer=_pool_init, initargs=(repo_root,)) as pool, tqdm(
            total=total_missing,
            desc=f"STOI кэш (×{nw} процессов, CPU)",
            unit="чанк",
            leave=True,
        ) as pbar:
            for batch in pool.imap_unordered(compute_stoi_group, payloads, chunksize=1):
                for k, v in batch:
                    self._stoi_cache[k] = v
                pbar.update(len(batch))

    def _persist_rich_cache(self) -> None:
        if not self.rich_cache_path:
            return
        for proc, orig, chunk in self.chunk_index:
            key = self._cache_key(proc, orig, chunk)
            noise, reverb = infer_noise_reverb_from_processed_path(proc)
            self._rich_entries[key] = {
                "degraded_path": str(proc.resolve()),
                "reference_path": str(orig.resolve()),
                "chunk": int(chunk),
                "stoi": float(self._stoi_cache[key]),
                "noise": bool(noise),
                "reverb": bool(reverb),
            }
        save_rich_cache(self.rich_cache_path, self._rich_entries)

    def __len__(self) -> int:
        return len(self.chunk_index)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        proc, orig, chunk = self.chunk_index[index]
        key = self._cache_key(proc, orig, chunk)
        stoi_v = self._stoi_cache[key]
        deg = self.waveform_preprocessor.process_degraded_file(proc, chunk)
        sample_id = f"{proc.name}#{chunk}"
        return {
            "waveform": deg.float(),
            "stoi_target": torch.tensor(stoi_v, dtype=torch.float32),
            "sample_id": sample_id,
        }

    def split_indices(
        self, train_ratio: float, val_ratio: float, test_ratio: float, seed: int
    ) -> Tuple[List[int], List[int], List[int]]:
        s = train_ratio + val_ratio + test_ratio
        if abs(s - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1")
        n = len(self)
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g).tolist()
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        train = perm[:n_train]
        val = perm[n_train : n_train + n_val]
        test = perm[n_train + n_val :]
        return train, val, test


def subset_by_indices(dataset: StoiPredictionDataset, indices: Sequence[int]) -> Subset:
    return Subset(dataset, list(indices))
