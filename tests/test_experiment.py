from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import red_herring
from red_herring import experiment
from red_herring.data import save_cache
from red_herring.experiment import (
    ExperimentConfig,
    config_from_args,
    headline_tables,
    iter_folds,
    make_model,
    parse_args,
    run_experiment,
    save_results,
    summarise,
)
from tests.conftest import make_spectra

FAST = ExperimentConfig(models=("lr",), seeds=1, folds=3, cal_splits=2, alphas=(0.2,))


def test_iter_folds_respects_groups_and_holds_out_species(spectra):
    folds = list(iter_folds(spectra, FAST))
    assert {f.unknown for f in folds} == {"SNA", "GUR", "TAR", "BCO"}
    groups = spectra.groups
    for f in folds:
        train, cal, test = (set(groups[i]) for i in (f.train, f.cal, f.test))
        assert train.isdisjoint(cal)
        assert train.isdisjoint(test)
        assert cal.isdisjoint(test)
        assert (f.data.y[f.unknown_idx] == -1).all()
        assert (f.data.y[np.r_[f.train, f.cal, f.test]] >= 0).all()
        assert f.meta["n_train"] == len(f.train)


def test_iter_folds_ungrouped_and_pairs(spectra):
    config = ExperimentConfig(seeds=1, folds=3, cal_splits=2, n_unknown=2, grouped=False)
    scenarios = {f.unknown for f in iter_folds(spectra, config)}
    assert len(scenarios) == 6
    assert "SNA+GUR" in scenarios


def test_iter_folds_skips_folds_missing_a_known_species():
    data = make_spectra(replicates=1)
    keep = [i for i, s in enumerate(data.species) if s != "SNA" or data.tissues[i] == "Frame"]
    data.X, data.y = data.X[keep], data.y[keep]
    data.individuals = [data.individuals[i] for i in keep]
    data.tissues = [data.tissues[i] for i in keep]
    data.sample_ids = [data.sample_ids[i] for i in keep]
    messages = []
    folds = list(iter_folds(data, FAST, log=messages.append))
    assert any("missing from proper-train" in m for m in messages)
    assert all(len(np.unique(f.data.y[f.train])) == len(f.data.classes) for f in folds)


def test_make_model(spectra):
    config = ExperimentConfig(epochs=2, feature="max")
    cnn = make_model("cnn", 3, config, spectra)
    assert cnn.seed == 3
    assert cnn.train_cfg.epochs == 2
    assert cnn.model_cfg.mz_max == spectra.X.shape[2]
    assert cnn.model_cfg.num_classes == 4
    assert make_model("svm", 0, config, spectra).feature == "max"


@pytest.fixture(scope="module")
def results():
    config = ExperimentConfig(
        models=("lr", "rf", "cnn"), seeds=1, folds=3, cal_splits=2, alphas=(0.1, 0.2), epochs=2
    )
    return run_experiment(make_spectra(), config, log=lambda _: None)


def test_run_experiment_metrics(results):
    metrics, _ = results
    assert set(metrics["model"]) == {"lr", "rf", "cnn"}
    assert set(metrics["unknown"]) == {"SNA", "GUR", "TAR", "BCO"}
    methods = set(metrics["method"])
    assert {"closed", "msp", "cp_lac", "cp_aps_mondrian", "cnd_msp"} <= methods
    assert {"knn", "mahalanobis", "energy"} <= set(metrics.loc[metrics.model == "cnn", "method"])
    assert "knn" not in set(metrics.loc[metrics.model == "rf", "method"])
    assert set(metrics["alpha"].dropna()) == {0.1, 0.2}
    assert metrics["value"].notna().all()


def test_run_experiment_predictions(results):
    _, predictions = results
    assert set(predictions["split"]) == {"cal", "test", "unknown"}
    assert (predictions.loc[predictions.split == "unknown", "y"] == -1).all()
    assert predictions["max_prob"].between(0, 1).all()
    assert "score_msp" in predictions


def test_summarise_and_headline_tables(results):
    metrics, _ = results
    summary = summarise(metrics)
    row = summary[
        (summary.model == "lr") & (summary.method == "closed") & (summary.metric == "bal_acc")
    ]
    assert (
        row["count"].item()
        == metrics.query("model == 'lr' and method == 'closed' and metric == 'bal_acc'").shape[0]
    )
    assert "unknown" in summarise(metrics, by_scenario=True)

    tables = headline_tables(summary, alpha=0.1)
    assert len(tables) == 4
    auroc = tables["Open-set detection AUROC"]
    assert set(auroc.index) == {"lr", "rf", "cnn"}
    assert "±" in auroc.loc["lr", "msp:auroc"]
    assert any("(alpha=0.1)" in title for title in tables)


def test_save_results(results, tmp_path):
    metrics, predictions = results
    summary = save_results(metrics, predictions, tmp_path / "out")
    for name in ("metrics.csv", "predictions.csv.gz", "summary.csv"):
        assert (tmp_path / "out" / name).exists()
    assert len(pd.read_csv(tmp_path / "out" / "summary.csv")) == len(summary)


def test_parse_args_and_config():
    args = parse_args(
        [
            "--models",
            "lr",
            "cnn",
            "--groups",
            "none",
            "--alphas",
            "0.1",
            "--n-unknown",
            "2",
            "--feature",
            "max",
        ]
    )
    config = config_from_args(args)
    assert config.models == ("lr", "cnn")
    assert config.grouped is False
    assert config.alphas == (0.1,)
    assert config.n_unknown == 2
    assert config.feature == "max"
    assert config_from_args(parse_args([])).grouped is True


def test_main_end_to_end(tmp_path, capsys, monkeypatch):
    cache = tmp_path / "chroma.npz"
    save_cache(make_spectra(), cache)
    out = tmp_path / "results"
    argv = [
        "--cache",
        str(cache),
        "--out",
        str(out),
        "--models",
        "lr",
        "--seeds",
        "1",
        "--folds",
        "3",
        "--cal-splits",
        "2",
        "--alphas",
        "0.1",
    ]
    experiment.main(argv)
    printed = capsys.readouterr().out
    assert "Open-set detection AUROC" in printed
    assert (out / "summary.csv").exists()

    monkeypatch.setattr("sys.argv", ["red-herring", *argv])
    red_herring.main()  # console-script entry point
    assert "Conformal novelty detection" in capsys.readouterr().out
