"""Model loading operations and validation."""

import json
from pathlib import Path

import torch

from panoseti_analysis.algorithms.cloud_detector import CloudDetection
from panoseti_analysis.config.models import ClassifierBundle, TrainingProvenance
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


def save_classifier(
    model: torch.nn.Module,
    bundle: ClassifierBundle,
    training_provenance: TrainingProvenance,
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    """Save a trained classifier as (state_dict .pt, ClassifierBundle .json, TrainingProvenance _provenance.json).

    The .json sidecar's checksum is computed from the written .pt file so that
    load_classifier's checksum verification passes unchanged.

    Returns:
        (pt_path, json_path, provenance_path)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = bundle.model_name

    pt_path = out_dir / f"{model_name}.pt"
    json_path = out_dir / f"{model_name}.json"
    provenance_path = out_dir / f"{model_name}_provenance.json"

    # Write state dict
    torch.save(model.state_dict(), str(pt_path))

    # Compute checksum of the .pt file and build a new bundle with the correct checksum
    actual_checksum = f"sha256:{compute_sha256(pt_path)}"
    # If bundle.checksum does not match the written file, rebuild with the correct checksum.
    # ClassifierBundle is frozen so we must construct a new instance.
    if bundle.checksum != actual_checksum:
        bundle = ClassifierBundle(
            model_name=bundle.model_name,
            model_version=bundle.model_version,
            checksum=actual_checksum,
            input_spec=bundle.input_spec,
        )

    json_path.write_text(bundle.model_dump_json(indent=2))

    # Also write output_checksum into training_provenance if it was None
    if training_provenance.output_checksum is None:
        training_provenance = TrainingProvenance(
            **{**training_provenance.model_dump(), "output_checksum": actual_checksum}
        )
    provenance_path.write_text(training_provenance.model_dump_json(indent=2))

    return pt_path, json_path, provenance_path
