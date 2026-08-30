"""SysML models distributed with CLARITY."""

from importlib.resources import files
from pathlib import Path


MODEL_DIRECTORIES = {
    "cruise": "cruise-controller-model",
    "mixing": "mixing-sysml-model",
    "thermostat": "thermostat",
}


def models_root() -> Path:
    """Return the installed directory containing the SysML models."""
    return Path(str(files(__package__)))


def model_path(name: str) -> Path:
    """Return the packaged SysML file for a named discrete model."""
    try:
        directory = MODEL_DIRECTORIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown model name: {name}") from exc
    return models_root() / directory / "model.sysml"
