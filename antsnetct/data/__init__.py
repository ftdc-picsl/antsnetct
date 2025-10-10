from __future__ import annotations

import csv
from importlib import resources

__all__ = ["get_label_definitions", "available_labelsets"]


def _normalize(name: str) -> str:
    return name if name.endswith(".tsv") else f"{name}.tsv"


def get_label_definitions_path(name: str) -> str:
    """Return the full path to antsnetct/data/<name>.tsv."""
    filename = _normalize(name)
    path = resources.files('antsnetct').joinpath('data', 'label_definitions', filename)
    if not path.is_file():
        raise FileNotFoundError(f"antsnetct/data/label_definitions/{filename} not found")
    return str(path)


def available_labelsets() -> list[str]:
    """List *.tsv files (without '.tsv') present in antsnetct/data."""
    root = resources.files('antsnetct').joinpath('data', 'label_definitions')
    if not root.is_dir():
        return []
    return sorted(
        p.name[:-4]
        for p in root.iterdir()
        if p.is_file() and p.name.lower().endswith('.tsv') and not p.name.startswith(('.', '_'))
    )
