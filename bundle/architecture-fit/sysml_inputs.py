"""Discover and describe controller models directly from SysML files."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from sysml_parser import SysMLParser


BOOLEAN_TYPES = {"bool", "boolean"}
REAL_TYPES = {"real", "float", "double"}


@dataclass(frozen=True)
class SysMLInput:
    key: str
    name: str
    path: Path
    sha256: str
    action_kind: str
    neural_action: str
    input_parameters: tuple[tuple[str, str], ...]
    completion_parameters: tuple[str, ...]
    output_parameters: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict:
        row = asdict(self)
        row["path"] = str(self.path)
        row["input_parameters"] = [list(item) for item in self.input_parameters]
        row["completion_parameters"] = list(self.completion_parameters)
        row["output_parameters"] = [list(item) for item in self.output_parameters]
        return row


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_name(text: str) -> str:
    match = re.search(r"\bpackage\s+(?:'([^']+)'|([A-Za-z_]\w*))\s*\{", text)
    if not match:
        raise ValueError("SysML file has no package declaration")
    return match.group(1) or match.group(2)


def _key(name: str) -> str:
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", name)
    key = re.sub(r"[^A-Za-z0-9]+", "-", separated).strip("-").lower()
    if not key:
        raise ValueError(f"SysML package name cannot form an output key: {name!r}")
    return key


def inspect_sysml(path: str | Path) -> SysMLInput:
    model_path = Path(path).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    text = model_path.read_text(encoding="utf-8")
    parser = SysMLParser(str(model_path))
    parser.parse()

    neural_defs = [
        action
        for part in parser.part_defs.values()
        for action in part.action_defs
        if "Neural" in action.metadata
    ]
    if len(neural_defs) != 1:
        raise ValueError(
            f"expected exactly one #Neural action in {model_path}, found {len(neural_defs)}"
        )
    neural = neural_defs[0]
    outputs = tuple((item.name, item.type_name) for item in neural.out_params)
    if not outputs:
        raise ValueError(f"#Neural action has no outputs in {model_path}")
    output_types = {(type_name or "").lower() for _name, type_name in outputs}
    if output_types <= BOOLEAN_TYPES:
        action_kind = "discrete"
    elif output_types <= REAL_TYPES:
        action_kind = "continuous"
    else:
        raise ValueError(
            f"#Neural outputs must be all Boolean or all real-valued in {model_path}: "
            f"{sorted(output_types)}"
        )

    name = _package_name(text)
    completion_parameters = tuple(
        item.name for item in neural.in_params if "Completion" in item.metadata
    )
    if len(completion_parameters) != 1:
        raise ValueError(
            f"#Neural action must mark exactly one input #Completion in {model_path}"
        )
    return SysMLInput(
        key=_key(name),
        name=name,
        path=model_path,
        sha256=_file_hash(model_path),
        action_kind=action_kind,
        neural_action=neural.name,
        input_parameters=tuple(
            (item.name, item.type_name) for item in neural.in_params
        ),
        completion_parameters=completion_parameters,
        output_parameters=outputs,
    )


def discover_sysml(
    supplied_paths: Iterable[str | Path],
    *,
    models_root: str | Path,
) -> list[SysMLInput]:
    paths = [Path(path).resolve() for path in supplied_paths]
    if not paths:
        root = Path(models_root).resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)
        paths = sorted(
            path.resolve()
            for path in root.rglob("*.sysml")
            if "#Neural" in path.read_text(encoding="utf-8")
        )
    if not paths:
        raise ValueError("no SysML files containing a #Neural action were found")

    models = sorted((inspect_sysml(path) for path in paths), key=lambda item: item.key)
    keys: dict[str, Path] = {}
    for model in models:
        previous = keys.get(model.key)
        if previous is not None:
            raise ValueError(
                f"SysML package names must be unique: {model.key!r} is used by "
                f"{previous} and {model.path}"
            )
        keys[model.key] = model.path
    return models
