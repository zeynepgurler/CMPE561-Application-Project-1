from pathlib import Path
from typing import List


def read_lines(path: str) -> List[str]:
    p = Path(path)
    return [ln.rstrip("\n") for ln in p.read_text(encoding="utf-8").splitlines()]


def read_tsv(path: str, sep: str = "\t"):
    rows = []
    for ln in read_lines(path):
        if not ln or ln.startswith("#"): continue
        parts = ln.split(sep)
        rows.append(parts)
    return rows


def write_lines(path: str, lines: List[str]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")