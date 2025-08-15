from __future__ import annotations

import csv
from functools import lru_cache
from importlib import resources
from typing import Dict, List

__all__ = ["get_label_definitions", "available_labelsets"]


def _normalize(name: str) -> str:
    return name if name.endswith(".tsv") else f"{name}.tsv"


@lru_cache(maxsize=None)
def get_label_definitions(name: str) -> Dict[int, str]:
    """
    Load antsnetct/data/<name>.tsv and return {index: name}.

    Assumes a BIDS-style TSV with header where the first two columns are:
        index<TAB>name
    Additional columns (if any) are ignored.
    """
    filename = _normalize(name)
    path = resources.files('antsnetct').joinpath('data', 'label_definitions', filename)
    if not path.is_file():
        raise FileNotFoundError(f"antsnetct/data/label_definitions/{filename} not found")

    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f, delimiter="\t")
        # Find the header (skip blank/comment lines)
        for row in reader:
            if not row or (row and row[0].lstrip().startswith("#")):
                continue
            header = [c.strip().lower() for c in row]
            break
        else:
            return {}

        if len(header) < 2 or header[0] != 'index' or header[1] != 'name':
            raise ValueError(
                f"{filename}: expected first two header columns to be 'index' and 'name', got {header[:2]}"
            )

        mapping: Dict[int, str] = {}
        for row in reader:
            if not row or (row and row[0].lstrip().startswith("#")):
                continue
            try:
                idx = int(row[0].strip())
            except ValueError as e:
                raise ValueError(f"{filename}: non-integer index in row {row}") from e
            lab = row[1].strip()
            mapping[idx] = lab

    return mapping


def available_labelsets() -> List[str]:
    """List *.tsv files (without '.tsv') present in antsnetct/data."""
    root = resources.files('antsnetct').joinpath('data', 'label_definitions')
    if not root.is_dir():
        return []
    return sorted(
        p.name[:-4]
        for p in root.iterdir()
        if p.is_file() and p.name.lower().endswith('.tsv') and not p.name.startswith(('.', '_'))
    )
