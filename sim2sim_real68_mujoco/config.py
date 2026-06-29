from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Sim2SimConfig:
    root: Path
    raw: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Sim2SimConfig":
        config_path = Path(path).resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        return cls(root=config_path.parent, raw=raw)

    def resolve_path(self, key: str) -> Path:
        value = self.raw[key]
        path = Path(value)
        return path if path.is_absolute() else (self.root / path).resolve()

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]
