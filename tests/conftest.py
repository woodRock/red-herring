"""Synthetic fixtures — tests never touch the private dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from red_herring.data import SPECIES, SpectraData, sqrt_l2

RUN_HEADER = """
RunHeaderInfo
dataset_id = 0, instrument_id = 0
first_scan = 1, last_scan = {n}, start_time = {start:.6f}, end_time = 44.991667
"""

SCAN = """
ScanHeader # {i}
position = {i}, start_mass= 50.000000, end_mass = 550.000000
start_time = {rt:.6f}, end_time = 0.000000, packet_type = 2
num_readings = {n_peaks}, integ_intens = 0.000000, data packet pos = 0
uScanCount = 0, PeakIntensity = 0.000000, PeakMass = 0.000000
DataPeaks
{peaks}
"""

PEAK = """Packet # {j}, intensity = {intensity:.6f}, mass/position = {mz:.6f}
saturated = 0, fragmented = 0, merged = 0
"""


def write_xcalibur(path: Path, scans: list[tuple[float, list[tuple[float, float]]]]) -> Path:
    """Write a minimal Xcalibur text dump: scans = [(rt, [(mz, intensity), ...]), ...]."""
    text = RUN_HEADER.format(n=len(scans), start=scans[0][0] if scans else 0.0)
    for i, (rt, peaks) in enumerate(scans, start=1):
        peak_text = "".join(
            PEAK.format(j=j, intensity=inten, mz=mz) for j, (mz, inten) in enumerate(peaks)
        )
        text += SCAN.format(i=i, rt=rt, n_peaks=len(peaks), peaks=peak_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


INDIVIDUALS = {"SNA": ["SNA1"], "GUR": ["GUR2"], "TAR": ["TAR3"], "BCO": ["BCO4", "BCO5"]}
TISSUES = ["Frame", "Head", "Liver", "Skin", "Gonad", "Rest of Guts"]


def make_spectra(
    n_bins: int = 12, mz: int = 24, replicates: int = 2, noise: float = 0.05, seed: int = 0
) -> SpectraData:
    """Species with distinct (rt, m/z) peaks plus noise, laid out like the real data."""
    rng = np.random.default_rng(seed)
    X, y, individuals, tissues, sample_ids = [], [], [], [], []
    for label, species in enumerate(SPECIES):
        signature = np.zeros((n_bins, mz), dtype=np.float32)
        signature[2 + 2 * label, 3 + 4 * label : 6 + 4 * label] = 1.0
        for individual in INDIVIDUALS[species]:
            for tissue in TISSUES:
                for rep in range(replicates):
                    x = signature + noise * rng.random((n_bins, mz), dtype=np.float32)
                    X.append(sqrt_l2(x))
                    y.append(label)
                    individuals.append(individual)
                    tissues.append(tissue)
                    sample_ids.append(f"{individual} {tissue} {'AB'[rep]}")
    return SpectraData(
        X=np.stack(X).astype(np.float32),
        y=np.array(y, dtype=np.int64),
        individuals=individuals,
        tissues=tissues,
        sample_ids=sample_ids,
    )


@pytest.fixture
def spectra() -> SpectraData:
    return make_spectra()


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """A miniature raw-data tree with one file per species plus files to be skipped."""
    root = tmp_path / "raw"
    files = {
        "SNA1/1 - SNA1 Head A.txt": 60.0,
        "GUR2/2 - GUR2 ROG B.txt": 70.0,
        "TAR/3 - TAR3 Liver Female A.txt": 80.0,
        "BCO4/4 - BCO4 Skin 2.txt": 90.0,
    }
    for rel, mz in files.items():
        write_xcalibur(root / rel, [(5.5, [(mz, 100.0)]), (20.0, [(mz + 1, 50.0)])])
    write_xcalibur(
        root / "BCO4/.ipynb_checkpoints/4 - BCO4 Skin 2-checkpoint.txt", [(5.5, [(1.0, 1.0)])]
    )
    write_xcalibur(root / "BCO4/notes.txt", [(5.5, [(1.0, 1.0)])])
    write_xcalibur(root / "OTHER/9 - XYZ9 Head A.txt", [(5.5, [(1.0, 1.0)])])
    return root
