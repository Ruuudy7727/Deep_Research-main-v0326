"""Single-source ablation configuration and runtime trace for paper experiments."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass
from typing import Any


VALID_VARIANTS = (
    "full",
    "wo_routing",
    "wo_bm25",
    "wo_schema",
    "wo_images",
    "wo_multi_agent",
)


@dataclass(frozen=True)
class AblationConfig:
    variant_name: str = "full"
    disable_bm25: bool = False
    bm25_alpha: float = 0.6
    disable_schema_constraint: bool = False
    disable_image_metadata: bool = False
    force_always_deep: bool = False
    force_always_complex: bool = False
    disable_multi_agent: bool = False
    # Kept for compatibility with older call sites. V2 does not expose this ablation.
    disable_tot: bool = False

    @classmethod
    def for_variant(cls, variant: str) -> "AblationConfig":
        name = str(variant or "full").strip().lower()
        if name not in VALID_VARIANTS:
            raise ValueError(
                f"Unsupported ABLATION_VARIANT={name!r}; "
                f"expected one of {', '.join(VALID_VARIANTS)}"
            )
        flags: dict[str, Any] = {"variant_name": name}
        if name == "wo_routing":
            flags.update(force_always_deep=True, force_always_complex=True)
        elif name == "wo_bm25":
            flags.update(disable_bm25=True, bm25_alpha=0.0)
        elif name == "wo_schema":
            flags.update(disable_schema_constraint=True)
        elif name == "wo_images":
            flags.update(disable_image_metadata=True)
        elif name == "wo_multi_agent":
            flags.update(disable_multi_agent=True)
        return cls(**flags)

    @classmethod
    def from_env(cls) -> "AblationConfig":
        return cls.for_variant(os.getenv("ABLATION_VARIANT", "full"))

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.public_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


_CONFIG: AblationConfig | None = None
_TRACE_LOCK = threading.Lock()
_TRACE: dict[str, Any] = {}


def get_ablation_config() -> AblationConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = AblationConfig.from_env()
        print(
            "[Ablation V2] "
            + json.dumps(
                {"config": _CONFIG.public_dict(), "fingerprint": _CONFIG.fingerprint},
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )
    return _CONFIG


def reset_ablation_config() -> None:
    global _CONFIG
    _CONFIG = None
    reset_ablation_trace()


def new_ablation_trace() -> dict[str, Any]:
    cfg = get_ablation_config()
    return {
        "variant": cfg.variant_name,
        "config_fingerprint": cfg.fingerprint,
        "execution_path": None,
        "executed_nodes": [],
        "dense_candidate_count": 0,
        "bm25_candidate_count": 0,
        "bm25_call_count": 0,
        "retrieved_image_count": 0,
        "injected_image_count": 0,
        "sql_mode": "direct_text2sql" if cfg.disable_schema_constraint else "plan_sanitize_build",
        "sanitizer_call_count": 0,
        "researcher_call_count": 0,
    }


def reset_ablation_trace() -> None:
    with _TRACE_LOCK:
        _TRACE.clear()
        _TRACE.update(new_ablation_trace())


def update_ablation_trace(**values: Any) -> None:
    with _TRACE_LOCK:
        if not _TRACE:
            _TRACE.update(new_ablation_trace())
        for key, value in values.items():
            if key.endswith("_increment"):
                target = key[: -len("_increment")]
                _TRACE[target] = int(_TRACE.get(target, 0) or 0) + int(value or 0)
            elif key == "executed_node":
                nodes = _TRACE.setdefault("executed_nodes", [])
                if value and value not in nodes:
                    nodes.append(value)
            else:
                _TRACE[key] = value


def get_ablation_trace() -> dict[str, Any]:
    with _TRACE_LOCK:
        if not _TRACE:
            _TRACE.update(new_ablation_trace())
        return json.loads(json.dumps(_TRACE, ensure_ascii=False, default=str))
