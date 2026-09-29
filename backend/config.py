"""Backend settings. Everything is overridable by environment variable so nothing is hard-coded."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_vehicle_model() -> str | None:
    """SAFESIGHT_VEHICLE_MODEL if set, otherwise the fine-tuned weights named in configs/detection.yaml."""
    if os.environ.get("SAFESIGHT_VEHICLE_MODEL"):
        return os.environ["SAFESIGHT_VEHICLE_MODEL"]
    try:
        import yaml

        with open(REPO_ROOT / "configs" / "detection.yaml") as f:
            path = REPO_ROOT / yaml.safe_load(f)["model_name"]
    except (OSError, KeyError, TypeError):
        return None
    return str(path) if path.exists() else None


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("SAFESIGHT_DATA_DIR", REPO_ROOT / "outputs" / "backend")))
    person_model: str = field(default_factory=lambda: os.environ.get("SAFESIGHT_PERSON_MODEL", "yolov8n.pt"))
    vehicle_model: str | None = field(default_factory=_default_vehicle_model)
    confidence_threshold: float = field(default_factory=lambda: float(os.environ.get("SAFESIGHT_CONFIDENCE", "0.35")))
    max_upload_mb: int = field(default_factory=lambda: int(os.environ.get("SAFESIGHT_MAX_UPLOAD_MB", "500")))

    @property
    def db_path(self) -> Path:
        return self.data_dir / "safesight.db"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def annotated_dir(self) -> Path:
        return self.data_dir / "annotated"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.uploads_dir, self.annotated_dir):
            d.mkdir(parents=True, exist_ok=True)
