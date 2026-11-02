"""Config loading: configs/base.yaml <- experiment yaml <- the `modes.<mode>` overlay (screen / debug / infer).

Every value the pipeline uses comes from here, so `config_resolved.yaml` in the experiment folder is a
complete record of the run (coding-style.md §2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """dict with attribute access: cfg.model.name == cfg["model"]["name"]."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def save(self, path: Path) -> None:
        path.write_text(yaml.safe_dump(dict(self), allow_unicode=True, sort_keys=False), encoding="utf-8")


def _deep_update(base: dict, overlay: dict) -> dict:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as f:
        return yaml.safe_load(f) or {}


def load_config(path: Path, mode: str = "full") -> Config:
    base_path = path.parent / "base.yaml"
    cfg = _load_yaml(base_path)
    if path.resolve() != base_path.resolve():
        _deep_update(cfg, _load_yaml(path))
    _deep_update(cfg, cfg.get("modes", {}).get(mode, {}))
    cfg["mode"] = mode
    project_root = path.parent.parent
    cfg["config_file"] = path.relative_to(project_root).as_posix() if path.is_relative_to(project_root) else path.as_posix()
    return Config(cfg)
