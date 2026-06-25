#!/usr/bin/env bash
# Run the full ablation matrix on a server with DB + dependencies.
# Usage (from repo root):
#   bash eval/run_on_server.sh
#   bash eval/run_on_server.sh --smoke   # --limit 1 per dataset

set -euo pipefail

BASE_URL="${EVAL_BASE_URL:-http://127.0.0.1:50221}"
OUT_ROOT="${EVAL_OUT_ROOT:-eval_outputs}"
LIMIT=""
if [[ "${1:-}" == "--smoke" ]]; then
  LIMIT="--limit 1"
fi

echo "== Step 0: start server_plus.py in another terminal, or use run_ablation_with_server.py per variant =="
echo "Base URL: ${BASE_URL}"
echo "Output root: ${OUT_ROOT}"

echo ""
echo "== Step 1: Ours-Full baseline (default env, no ABLATION_* flags) =="
python eval/run_ablation_suite.py \
  --base-url "${BASE_URL}" \
  --out-root "${OUT_ROOT}" \
  --variants full \
  ${LIMIT}

variants=(wo_routing wo_bm25 wo_schema wo_images wo_tot)
for variant in "${variants[@]}"; do
  echo ""
  echo "== Step: ${variant} — restart server with env from eval/ablation_variants.json =="
  python - <<PY
import json
from pathlib import Path
cfg = json.loads(Path("eval/ablation_variants.json").read_text(encoding="utf-8"))
env = cfg["${variant}"].get("env") or {}
print("export these before starting server_plus.py:")
for k, v in env.items():
    print(f"  export {k}={v}")
PY
  read -r -p "Press Enter after server restarted with the env above..."
  python eval/run_ablation_suite.py \
    --base-url "${BASE_URL}" \
    --out-root "${OUT_ROOT}" \
    --variants "${variant}" \
    ${LIMIT}
done

echo ""
echo "== Step 3: compare + Table III =="
python eval/compare_ablation_results.py --root "${OUT_ROOT}"

echo ""
echo "Done. See:"
echo "  ${OUT_ROOT}/analysis/ablation_comparison.json"
echo "  eval/paper/table_iii.md"
echo "  eval/paper/section_4_10_failures.md"
