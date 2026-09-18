"""Tests for the controlled LightEdgeDet ablation harness."""

import argparse
import csv
import subprocess
from pathlib import Path

import pytest

from scripts import ablate

CONFIG = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/lightedgedet_nano.yaml"


def study_args(tmp_path, **overrides):
    """Return a minimal study invocation with a disposable output directory."""
    values = {
        "config": str(CONFIG),
        "data": "coco8.yaml",
        "epochs": 3,
        "imgsz": 128,
        "batch": 4,
        "workers": 0,
        "device": "cpu",
        "seed": 0,
        "project": str(tmp_path / "study"),
        "build_only": False,
        "ablate": None,
        "set": None,
        "quiet": True,
    }
    return argparse.Namespace(**(values | overrides))


def test_default_variants_change_only_the_requested_components(tmp_path):
    """Default study contains the full model and three nonempty one-factor ablations."""
    base, variants = ablate.variants_from_args(study_args(tmp_path))
    assert variants == [("baseline", {}), *ablate.CORE_ABLATIONS]
    assert base["backbone_attn_stages"] == [0, 0, 0, 1, 1]
    assert variants[-1][1] == {"backbone_attn_stages": [0, 0, 0, 0, 0]}


@pytest.mark.parametrize("override", ["absent_key=False", "neck_use_cross_fusion=True"])
def test_unknown_or_noop_override_is_rejected(tmp_path, override):
    """A typo or unchanged value must not masquerade as an ablation."""
    with pytest.raises(ValueError):
        ablate.variants_from_args(study_args(tmp_path, ablate=[["bad", override]]))


def test_history_requires_every_epoch(tmp_path):
    """An incomplete training log cannot be treated as a completed experiment."""
    path = tmp_path / "results.csv"
    path.write_text("epoch,metrics/mAP50-95(B)\n1,0.1\n2,0.2\n3,0.3\n")
    assert ablate.read_history(path, 3) == (3, 3)
    with pytest.raises(RuntimeError, match="logged 3/4"):
        ablate.read_history(path, 4)


def test_training_requires_fixed_batch(tmp_path):
    """Do not allow auto-batch to change the effective recipe across variants."""
    with pytest.raises(ValueError, match="fixed positive integer --batch"):
        ablate.run_study(study_args(tmp_path, batch=None))
    assert not (tmp_path / "study").exists()


def test_failed_worker_keeps_metrics_blank(tmp_path, monkeypatch):
    """A worker crash is recorded as failure, never as a partial metric row."""
    monkeypatch.setattr(ablate, "profile_model", lambda config, imgsz: (1.0, 0.9, 2.0))

    def fail_worker(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(ablate.subprocess, "run", fail_worker)
    assert ablate.run_study(study_args(tmp_path, ablate=[["no_psa", "backbone_attn_stages=[0,0,0,0,0]"]])) == 1
    with (tmp_path / "study/ablation_results.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert all(row["status"] == "failed" and row["metrics/mAP50-95(B)"] == "" for row in rows)
