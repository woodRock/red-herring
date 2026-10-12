"""
NZ fish species GC-MS data — raw Xcalibur text dumps → 2D chromatograms.

Processing follows chroma-dcnn (scripts/preprocess_fish_oil_chroma.py):
  1. Parse every scan's centroided peaks (mass, intensity) and retention time
  2. Map m/z to a 1000-bin nominal-mass grid (rounded to the nearest Da)
  3. Divide the RT axis into N equal-width windows (~24 scans each at N=200)
  4. In each window keep the highest-TIC scan (zero vector if all-zero)
  5. sqrt + L2 normalise each selected scan
  → float32 [N, 1000] per sample

Unlike the chroma-dcnn CSVs, runs here start at either 3.0 or 5.0 min
(5040 vs 4800 scans), so windows are defined on a shared RT range rather
than on scan index; otherwise the same RT bin would mean different times.

Open-set formulation: species can be held out as unknown. Known species are
relabelled 0..K-1 and held-out species get label -1 (see `open_set_split`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

dir_path = Path("data/raw-subsetted-CVSs")
data_directories = ["BCO4", "GUR2", "SNA1", "TAR"]

MZ_MAX = 1000
N_BINS = 200
RT_RANGE = (5.0, 45.0)  # minutes; common to every run

SPECIES = ["SNA", "GUR", "TAR", "BCO"]
SPECIES_LABEL = {s: i for i, s in enumerate(SPECIES)}
UNKNOWN = -1

# "4 - BCO4 Head A.txt", "2 - GUR2  Frame A.txt", "6 - BCO6 Liver Male.txt"
_FNAME_RE = re.compile(r"\d+ - ([A-Z]{3})(\d+)\s+(.+)\.txt$", re.IGNORECASE)
_SCAN_RE = re.compile(r"ScanHeader # \d+\n[^\n]*\nstart_time = ([\d.]+)")
_PEAK_RE = re.compile(r"intensity = ([\d.]+), mass/position = ([\d.]+)")
_TISSUE_ALIASES = {"ROG": "Rest of Guts"}


@dataclass
class SpectraData:
    X: np.ndarray                 # [N, n_bins, MZ_MAX] float32
    y: np.ndarray                 # [N] int64 species label (UNKNOWN for open-set held out)
    individuals: list[str]        # biological individual, e.g. "BCO4" — use as CV groups
    tissues: list[str]            # body part, e.g. "Rest of Guts"
    sample_ids: list[str]
    classes: list[str] = field(default_factory=lambda: list(SPECIES))

    def __len__(self) -> int:
        return len(self.y)

    @property
    def species(self) -> list[str]:
        return [self.classes[c] if c != UNKNOWN else "unknown" for c in self.y]


def _sqrt_l2(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.sqrt(np.maximum(arr, 0.0))
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / (norms + eps)


def read_scans(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse an Xcalibur text dump into per-scan spectra.

    Returns
    -------
    rt      : [n_scans] float32 retention times (minutes)
    spectra : [n_scans, MZ_MAX] float32 intensities on the nominal-mass grid
    """
    text = file_path.read_text(encoding="utf-8", errors="ignore")

    scans = list(_SCAN_RE.finditer(text))
    rt = np.array([float(m.group(1)) for m in scans], dtype=np.float32)
    scan_starts = np.array([m.start() for m in scans])

    peaks = list(_PEAK_RE.finditer(text))
    spectra = np.zeros((len(scans), MZ_MAX), dtype=np.float32)
    if not peaks:
        return rt, spectra

    peak_pos = np.array([m.start() for m in peaks])
    intensity = np.array([float(m.group(1)) for m in peaks], dtype=np.float32)
    mz = np.rint([float(m.group(2)) for m in peaks]).astype(np.int64)
    scan_idx = np.searchsorted(scan_starts, peak_pos, side="right") - 1

    keep = (mz >= 0) & (mz < MZ_MAX) & (scan_idx >= 0)
    np.add.at(spectra, (scan_idx[keep], mz[keep]), intensity[keep])
    return rt, spectra


def _bin_chromatogram(
    rt: np.ndarray, spectra: np.ndarray, n_bins: int, rt_range: tuple[float, float]
) -> np.ndarray:
    """
    Divide rt_range into n_bins equal windows; keep the highest-TIC scan per window.

    Returns [n_bins, MZ_MAX] — zero rows where the window has no signal.
    """
    rt_min, rt_max = rt_range
    tic = spectra.sum(axis=1)
    bin_idx = np.floor((rt - rt_min) / (rt_max - rt_min) * n_bins).astype(np.int64)

    result = np.zeros((n_bins, spectra.shape[1]), dtype=np.float32)
    for i in range(n_bins):
        in_bin = np.flatnonzero(bin_idx == i)
        if len(in_bin) == 0:
            continue
        best = in_bin[np.argmax(tic[in_bin])]
        result[i] = spectra[best]
    return result


def process_file(
    file_path: Path, n_bins: int = N_BINS, rt_range: tuple[float, float] = RT_RANGE
) -> np.ndarray:
    rt, spectra = read_scans(file_path)
    chroma = _bin_chromatogram(rt, spectra, n_bins, rt_range)  # [n_bins, MZ_MAX]
    return _sqrt_l2(chroma)                                     # normalise each bin


def parse_filename(name: str) -> tuple[str, str, str] | None:
    """'4 - BCO4 Liver Male B.txt' → ('BCO', 'BCO4', 'Liver Male')."""
    m = _FNAME_RE.match(name)
    if m is None:
        return None
    species, number, rest = m.group(1).upper(), m.group(2), m.group(3).split()
    # Trailing replicate marker: "A", "B", "C", or "2" (SNA1 Head 2).
    if len(rest) > 1 and len(rest[-1]) == 1:
        rest = rest[:-1]
    tissue = " ".join(rest)
    return species, f"{species}{number}", _TISSUE_ALIASES.get(tissue, tissue)


def load_raw(
    path: str | Path = dir_path,
    n_bins: int = N_BINS,
    rt_range: tuple[float, float] = RT_RANGE,
    cache: str | Path | None = None,
) -> SpectraData:
    """
    Load every sample under `path`/{BCO4,GUR2,SNA1,TAR} as a 2D chromatogram.

    Parsing the text dumps is slow (~1.6 GB), so pass `cache` (an .npz path)
    to save the result on first run and reload it afterwards.
    """
    if cache is not None and Path(cache).exists():
        return _load_cache(Path(cache))

    root = Path(path)
    X, y, individuals, tissues, sample_ids = [], [], [], [], []
    for directory in sorted(root.iterdir()):
        if not (directory.is_dir() and directory.name in data_directories):
            continue
        # Not recursive: skips .ipynb_checkpoints duplicates.
        for file_path in sorted(directory.glob("*.txt")):
            parsed = parse_filename(file_path.name)
            if parsed is None:
                print(f"Skipping unrecognised file: {file_path}")
                continue
            species, individual, tissue = parsed
            X.append(process_file(file_path, n_bins, rt_range))
            y.append(SPECIES_LABEL[species])
            individuals.append(individual)
            tissues.append(tissue)
            sample_ids.append(file_path.stem)

    data = SpectraData(
        X=np.stack(X),
        y=np.array(y, dtype=np.int64),
        individuals=individuals,
        tissues=tissues,
        sample_ids=sample_ids,
    )
    if cache is not None:
        _save_cache(data, Path(cache))
    return data


def open_set_split(data: SpectraData, unknown: list[str]) -> SpectraData:
    """
    Hold out `unknown` species: they are labelled UNKNOWN (-1), and the
    remaining known species are relabelled 0..K-1 in SPECIES order.
    """
    unknown = {s.upper() for s in unknown}
    if not unknown <= set(data.classes):
        raise ValueError(f"Unknown species must be drawn from {data.classes}, got {sorted(unknown)}")
    known = [s for s in data.classes if s not in unknown]
    remap = {data.classes.index(s): i for i, s in enumerate(known)}
    y = np.array([remap.get(int(c), UNKNOWN) for c in data.y], dtype=np.int64)
    return SpectraData(
        X=data.X,
        y=y,
        individuals=data.individuals,
        tissues=data.tissues,
        sample_ids=data.sample_ids,
        classes=known,
    )


def _save_cache(data: SpectraData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=data.X,
        y=data.y,
        individuals=np.array(data.individuals),
        tissues=np.array(data.tissues),
        sample_ids=np.array(data.sample_ids),
        classes=np.array(data.classes),
    )


def _load_cache(path: Path) -> SpectraData:
    f = np.load(path)
    return SpectraData(
        X=f["X"],
        y=f["y"],
        individuals=f["individuals"].tolist(),
        tissues=f["tissues"].tolist(),
        sample_ids=f["sample_ids"].tolist(),
        classes=f["classes"].tolist(),
    )
