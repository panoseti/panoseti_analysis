"""Model loading operations and validation."""

import json
from pathlib import Path

import torch

from panoseti_analysis.algorithms.cloud_detector import CloudDetection
from panoseti_analysis.config.models import ClassifierBundle
from panoseti_analysis.io.checksum import compute_sha256


def load_classifier(model_path: Path) -> tuple[torch.nn.Module, ClassifierBundle]:
    """Load a PyTorch model and its metadata sidecar.

    Verifies the model file's SHA256 checksum against the sidecar before loading.

    Args:
        model_path: Path to the .pt or .pth file.

    Returns:
        The loaded PyTorch module and its validated ClassifierBundle.
    """
    json_path = model_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing model metadata sidecar: {json_path}")

    with json_path.open() as f:
        data = json.load(f)

    bundle = ClassifierBundle(**data)

    # Verify checksum
    actual_sha = f"sha256:{compute_sha256(model_path)}"
    if actual_sha != bundle.checksum:
        raise ValueError(f"Model checksum mismatch for {model_path}. Expected {bundle.checksum}, got {actual_sha}")

    # The .pt file is a state_dict (OrderedDict)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model = CloudDetection()
    model.load_state_dict(state_dict)

    return model, bundle
