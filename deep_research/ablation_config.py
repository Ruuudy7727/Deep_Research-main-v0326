"""Central ablation toggles for paper experiments.

All flags are controlled via environment variables so each ablation run can
restart the service with a single changed knob. See eval/ablation_variants.json.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class AblationConfig:
    """Snapshot of active ablation settings."""

    disable_bm25: bool = False
    bm25_alpha: float = 0.6
    disable_schema_constraint: bool = False
    disable_image_metadata: bool = False
    disable_tot: bool = False
    force_always_deep: bool = False
    force_always_complex: bool = False
    disable_multi_agent: bool = False
    variant_name: str = "full"

    @classmethod
    def from_env(cls) -> "AblationConfig":
        disable_bm25 = _env_bool("ABLATION_DISABLE_BM25")
        alpha = _env_float("BM25_ALPHA", 0.6 if not disable_bm25 else 0.0)
        if disable_bm25:
            alpha = 0.0
        return cls(
            disable_bm25=disable_bm25,
            bm25_alpha=alpha,
            disable_schema_constraint=_env_bool("ABLATION_DISABLE_SCHEMA"),
            disable_image_metadata=_env_bool("ABLATION_DISABLE_IMAGES"),
            disable_tot=_env_bool("ABLATION_DISABLE_TOT"),
            force_always_deep=_env_bool("ABLATION_FORCE_ALWAYS_DEEP"),
            force_always_complex=_env_bool("ABLATION_FORCE_ALWAYS_COMPLEX"),
            disable_multi_agent=_env_bool("ABLATION_DISABLE_MULTI_AGENT"),
            variant_name=os.getenv("ABLATION_VARIANT", "full").strip() or "full",
        )


_CONFIG: AblationConfig | None = None


def get_ablation_config() -> AblationConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = AblationConfig.from_env()
    return _CONFIG


def reset_ablation_config() -> None:
    """Re-read environment (mainly for tests)."""
    global _CONFIG
    _CONFIG = None
