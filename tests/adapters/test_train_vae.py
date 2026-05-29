"""Adapter tests for pa-train-vae (BetaVAE training on PH feature cache)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import xarray as xr
import yaml

from panoseti_analysis.algorithms.ph_vae import BetaVAE
from panoseti_analysis.config.models import ClassifierBundle, TrainingProvenance
from panoseti_analysis.config.versions import PANOSETI_ANALYSIS_STORAGE_VERSION
from panoseti_analysis.io.models import load_vae, save_classifier
from panoseti_analysis.io.provenance import now_utc

_RECIPE = Path(__file__).parents[2] / "recipes" / "vae_train_v1.yml"
_EPOCH_NS = 1_700_000_000_000_000_000


def _make_ph_feature_cache(tmp_path: Path, *, n_samples: int = 20) -> Path:
    """Write a minimal ph_features Zarr store directly (bypasses pa-prep-ph)."""
    rng = np.random.default_rng(7)
    X = rng.standard_normal((n_samples, 1, 16, 16)).astype(np.float32)
    t = _EPOCH_NS + np.arange(n_samples, dtype=np.int64) * 1_000_000
    module_arr = np.array(["1"] * n_samples, dtype=str)

    store_path = tmp_path / "ph_features.zarr"
    ds = xr.Dataset(
        data_vars={
            "X": (["sample", "channel", "H", "W"], X),
            "t_ns": (["sample"], t),
            "module": (["sample"], module_arr),
        },
        attrs={
            "recipe_name": "test_recipe",
            "recipe_hash": "sha256:test",
            "data_level": "ph_features",
        },
    )
    ds.to_zarr(str(store_path), zarr_format=3)
    return store_path


def _write_test_recipe(tmp_path: Path) -> Path:
    """Write a minimal BetaVAE training recipe for fast CPU tests."""
    recipe = {
        "name": "vae_test_tiny",
        "latent_dim": 4,
        "hidden_dim": 8,
        "beta": 4.0e-9,
        "sparsity_weight": 0.0,
        "hyperparams": {
            "lr": 0.001,
            "batch_size": 4,
            "epochs": 2,
            "weight_decay": 0.0,
        },
        "scaling": {
            "num_workers": 1,
            "accelerator_type": None,
        },
    }
    recipe_path = tmp_path / "vae_test.yml"
    recipe_path.write_text(yaml.safe_dump(recipe))
    return recipe_path


# ── unit test: save + load VAE roundtrip (no Ray) ─────────────────────────────

def test_train_vae_save_load_roundtrip(tmp_path: Path) -> None:
    """save_classifier + load_vae must produce a consistent checkpoint."""
    model = BetaVAE(latent_dim=4, hidden_dim=8)

    bundle = ClassifierBundle(
        model_name="ph_vae",
        model_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        checksum="sha256:placeholder",
        input_spec={
            "channels": 1,
            "height": 16,
            "width": 16,
            "dtype": "float32",
            "latent_dim": 4,
            "hidden_dim": 8,
        },
    )
    prov = TrainingProvenance(
        step_name="train_vae",
        step_version=PANOSETI_ANALYSIS_STORAGE_VERSION,
        recipe_name="test",
        recipe_hash="sha256:abc",
        params={"latent_dim": 4, "hidden_dim": 8, "beta": 4e-9},
        input_checksums=["sha256:deadbeef"],
        timestamp_utc=now_utc(),
        metrics={"train_loss": 0.042},
    )

    pt_path, json_path, provenance_path = save_classifier(model, bundle, prov, tmp_path)

    # Files must exist
    assert pt_path.exists(), f".pt file missing: {pt_path}"
    assert json_path.exists(), f".json sidecar missing: {json_path}"
    assert provenance_path.exists(), f"_provenance.json missing: {provenance_path}"

    # .json must have a real sha256 checksum (not placeholder)
    import json as _json
    meta = _json.loads(json_path.read_text())
    assert meta["checksum"].startswith("sha256:")
    assert meta["checksum"] != "sha256:placeholder"

    # _provenance.json must parse as TrainingProvenance
    prov_loaded = TrainingProvenance.model_validate_json(provenance_path.read_text())
    assert prov_loaded.metrics.get("train_loss") == pytest.approx(0.042)

    # load_vae must reconstruct the model with correct architecture
    loaded_model, loaded_bundle = load_vae(pt_path, latent_dim=4, hidden_dim=8)
    assert isinstance(loaded_model, BetaVAE)
    # Verify forward pass shape still works
    x = torch.randn(2, 1, 16, 16)
    recon, mu, _logvar = loaded_model(x)
    assert recon.shape == (2, 1, 16, 16)
    assert mu.shape == (2, 4)

    # Bundle input_spec must carry latent_dim/hidden_dim
    assert loaded_bundle.input_spec.get("latent_dim") == 4
    assert loaded_bundle.input_spec.get("hidden_dim") == 8


# ── slow integration test: end-to-end via Ray Train ──────────────────────────

@pytest.mark.slow
def test_train_vae_checkpoint_roundtrip(tmp_path: Path) -> None:
    """End-to-end: feature cache → run_train_vae → pt file with TrainingProvenance."""
    feature_cache = _make_ph_feature_cache(tmp_path / "features", n_samples=20)
    recipe_path = _write_test_recipe(tmp_path)
    out_dir = tmp_path / "model_out"

    from panoseti_analysis.adapters.ray.train_vae import run_train_vae

    pt_path, json_path, provenance_path = run_train_vae(
        feature_cache, out_dir, recipe_path, launcher="standalone"
    )

    assert pt_path.exists(), f".pt file not written: {pt_path}"
    assert json_path.exists(), f".json sidecar not written: {json_path}"
    assert provenance_path.exists(), f"_provenance.json not written: {provenance_path}"

    prov = TrainingProvenance.model_validate_json(provenance_path.read_text())
    assert "train_loss" in prov.metrics, (
        f"'train_loss' missing from provenance metrics: {prov.metrics}"
    )
    assert isinstance(prov.metrics["train_loss"], float)
