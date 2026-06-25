"""Start server_plus.py with one validated V2 ablation variant."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LEGACY_KEYS = (
    "ABLATION_DISABLE_BM25",
    "ABLATION_DISABLE_SCHEMA",
    "ABLATION_DISABLE_IMAGES",
    "ABLATION_DISABLE_TOT",
    "ABLATION_FORCE_ALWAYS_DEEP",
    "ABLATION_FORCE_ALWAYS_COMPLEX",
    "ABLATION_DISABLE_MULTI_AGENT",
    "BM25_ALPHA",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()

    from deep_research.ablation_config import AblationConfig

    cfg = AblationConfig.for_variant(args.variant)
    env = os.environ.copy()
    for key in LEGACY_KEYS:
        env.pop(key, None)
    env["ABLATION_VARIANT"] = cfg.variant_name
    print(f"[serve_variant] variant={cfg.variant_name} fingerprint={cfg.fingerprint}", flush=True)
    return subprocess.call([sys.executable, str(ROOT / "server_plus.py")], cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
