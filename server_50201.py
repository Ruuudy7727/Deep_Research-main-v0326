"""Start an independent server_plus instance on port 50201."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

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
    parser = argparse.ArgumentParser(
        description="Start server_plus.py on port 50201 with one V2 ablation variant."
    )
    parser.add_argument("--variant", required=True)
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from deep_research.ablation_config import AblationConfig

    config = AblationConfig.for_variant(args.variant)
    env = os.environ.copy()
    for key in LEGACY_KEYS:
        env.pop(key, None)

    env["ABLATION_VARIANT"] = config.variant_name
    env["SERVER_PLUS_PORT"] = "50201"

    print(
        f"[server_50201] variant={config.variant_name} "
        f"fingerprint={config.fingerprint} port=50201",
        flush=True,
    )
    return subprocess.call(
        [sys.executable, str(ROOT / "server_plus.py")],
        cwd=ROOT,
        env=env,
    )


if __name__ == "__main__":
    raise SystemExit(main())
