from __future__ import annotations

import numpy as np
import pytest

from red_herring.data import (
    MZ_MAX,
    UNKNOWN,
    bin_chromatogram,
    load_raw,
    open_set_split,
    parse_filename,
    process_file,
    read_scans,
    sqrt_l2,
)
from tests.conftest import write_xcalibur


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("4 - BCO4 Head A.txt", ("BCO", "BCO4", "Head")),
        ("6 - BCO6 Liver Male B.txt", ("BCO", "BCO6", "Liver Male")),
        ("6 - BCO6 Liver Male.txt", ("BCO", "BCO6", "Liver Male")),
        ("2 - GUR2  Frame A.txt", ("GUR", "GUR2", "Frame")),
        ("2 - GUR2 ROG A.txt", ("GUR", "GUR2", "Rest of Guts")),
        ("1 - SNA1 Head 2.txt", ("SNA", "SNA1", "Head")),
        ("3 - TAR3 Rest of Guts B.txt", ("TAR", "TAR3", "Rest of Guts")),
        ("1 - SNA1 Skin.txt", ("SNA", "SNA1", "Skin")),
    ],
)
def test_parse_filename(name, expected):
    assert parse_filename(name) == expected


@pytest.mark.parametrize("name", ["notes.txt", "BCO4 Head A.txt", "4 - BCO4 Head A.csv"])
def test_parse_filename_rejects_other_files(name):
    assert parse_filename(name) is None


def test_sqrt_l2_gives_unit_rows_and_keeps_zero_rows():
    out = sqrt_l2(np.array([[4.0, 0.0, 9.0], [0.0, 0.0, 0.0], [-1.0, 1.0, 0.0]]))
    np.testing.assert_allclose(np.linalg.norm(out[[0, 2]], axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(out[1], 0.0)
    assert out[2, 0] == 0.0  # negatives clipped before sqrt


def test_read_scans_rounds_mass_and_sums_shared_bins(tmp_path):
    path = write_xcalibur(
        tmp_path / "run.txt",
        [
            (5.0, []),
            (5.1, [(121.65, 10.0), (122.4, 5.0), (122.55, 2.0)]),
            (5.2, [(1200.0, 7.0), (60.0, 3.0)]),  # out-of-range m/z dropped
        ],
    )
    rt, spectra = read_scans(path)
    np.testing.assert_allclose(rt, [5.0, 5.1, 5.2], rtol=1e-6)
    assert spectra.shape == (3, MZ_MAX)
    assert spectra[0].sum() == 0
    assert spectra[1, 122] == 15.0  # 121.65 and 122.4 both round to 122
    assert spectra[1, 123] == 2.0
    assert spectra[2, 60] == 3.0
    assert spectra[2].sum() == 3.0


def test_read_scans_without_peaks(tmp_path):
    rt, spectra = read_scans(write_xcalibur(tmp_path / "empty.txt", [(5.0, []), (5.1, [])]))
    assert len(rt) == 2
    assert not spectra.any()


def test_bin_chromatogram_keeps_highest_tic_scan_per_window():
    rt = np.array([0.0, 0.5, 1.2, 3.5, 9.0])
    spectra = np.array([[1, 0], [0, 5], [2, 2], [0, 1], [7, 7]], dtype=np.float32)
    out = bin_chromatogram(rt, spectra, n_bins=4, rt_range=(0.0, 4.0))
    np.testing.assert_array_equal(out, [[0, 5], [2, 2], [0, 0], [0, 1]])  # rt=9 is out of range


def test_process_file_aligns_runs_with_different_start_times(tmp_path):
    peak = [(100.0, 50.0)]
    early = write_xcalibur(tmp_path / "a.txt", [(3.0, [(80.0, 9.0)]), (10.0, peak)])
    late = write_xcalibur(tmp_path / "b.txt", [(5.0, []), (10.0, peak)])
    a, b = (process_file(p, n_bins=8, rt_range=(5.0, 45.0)) for p in (early, late))
    assert a.shape == (8, MZ_MAX)
    np.testing.assert_allclose(a, b)  # the scan before 5 min is ignored
    assert a[1, 100] == pytest.approx(1.0)


def test_load_raw_reads_species_directories_only(raw_dir, capsys):
    data = load_raw(raw_dir, n_bins=10)
    assert data.X.shape == (4, 10, MZ_MAX)
    assert data.sample_ids == [
        "4 - BCO4 Skin 2",
        "2 - GUR2 ROG B",
        "1 - SNA1 Head A",
        "3 - TAR3 Liver Female A",
    ]
    assert data.species == ["BCO", "GUR", "SNA", "TAR"]
    assert data.individuals == ["BCO4", "GUR2", "SNA1", "TAR3"]
    assert data.tissues == ["Skin", "Rest of Guts", "Head", "Liver Female"]
    assert "Skipping unrecognised file" in capsys.readouterr().out
    assert data.X[0, 0, 90] > 0  # BCO peak at m/z 90 in the first RT bin


def test_load_raw_cache_roundtrip(raw_dir, tmp_path):
    cache = tmp_path / "cache" / "chroma.npz"
    first = load_raw(raw_dir, n_bins=10, cache=cache)
    assert cache.exists()
    second = load_raw(tmp_path / "does-not-exist", cache=cache)  # served from cache
    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)
    assert first.sample_ids == second.sample_ids
    assert first.individuals == second.individuals
    assert first.tissues == second.tissues
    assert first.classes == second.classes


def test_groups_join_individual_and_tissue(spectra):
    groups = spectra.groups
    assert groups[0] == "SNA1|Frame"
    assert len(set(groups)) == len(spectra) // 2  # two replicates per group


def test_open_set_split_relabels_known_and_marks_unknown(spectra):
    osd = open_set_split(spectra, ["gur"])
    assert osd.classes == ["SNA", "TAR", "BCO"]
    original = np.array(spectra.species)
    assert (osd.y[original == "GUR"] == UNKNOWN).all()
    np.testing.assert_array_equal(osd.y[original == "SNA"], 0)
    np.testing.assert_array_equal(osd.y[original == "TAR"], 1)
    np.testing.assert_array_equal(osd.y[original == "BCO"], 2)
    assert osd.species.count("unknown") == (original == "GUR").sum()
    assert osd.X is spectra.X  # no copy of the chromatograms
    assert len(osd) == len(spectra)


def test_open_set_split_multiple_unknowns(spectra):
    osd = open_set_split(spectra, ["SNA", "BCO"])
    assert osd.classes == ["GUR", "TAR"]
    assert set(np.unique(osd.y)) == {UNKNOWN, 0, 1}


def test_open_set_split_rejects_unknown_species(spectra):
    with pytest.raises(ValueError, match="must be drawn from"):
        open_set_split(spectra, ["COD"])
