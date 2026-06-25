from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

from deep_research.ablation_config import (
    AblationConfig,
    get_ablation_trace,
    reset_ablation_trace,
    update_ablation_trace,
)

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "eval" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class AblationConfigTests(unittest.TestCase):
    def test_variants_are_single_source_and_stable(self):
        expected = {
            "full": {},
            "wo_routing": {"force_always_deep": True},
            "wo_bm25": {"disable_bm25": True, "bm25_alpha": 0.0},
            "wo_schema": {"disable_schema_constraint": True},
            "wo_images": {"disable_image_metadata": True},
            "wo_multi_agent": {"disable_multi_agent": True},
        }
        for name, flags in expected.items():
            cfg = AblationConfig.for_variant(name)
            self.assertEqual(cfg.variant_name, name)
            self.assertEqual(len(cfg.fingerprint), 16)
            for key, value in flags.items():
                self.assertEqual(getattr(cfg, key), value)

    def test_invalid_variant_fails(self):
        with self.assertRaises(ValueError):
            AblationConfig.for_variant("wo_fake")

    def test_trace_counters(self):
        old = os.environ.get("ABLATION_VARIANT")
        os.environ["ABLATION_VARIANT"] = "full"
        try:
            reset_ablation_trace()
            update_ablation_trace(bm25_call_count_increment=2, executed_node="node_a")
            update_ablation_trace(executed_node="node_a")
            trace = get_ablation_trace()
            self.assertEqual(trace["bm25_call_count"], 2)
            self.assertEqual(trace["executed_nodes"], ["node_a"])
        finally:
            if old is None:
                os.environ.pop("ABLATION_VARIANT", None)
            else:
                os.environ["ABLATION_VARIANT"] = old


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = load_script("run_experiment")
        cls.analyzer = load_script("analyze_experiment")

    def test_dataset_has_40_unique_rows(self):
        rows = []
        for line in (ROOT / "eval" / "datasets" / "v2_ablation_40.jsonl").read_text(
            encoding="utf-8"
        ).splitlines():
            rows.append(json.loads(line))
        self.assertEqual(len(rows), 40)
        self.assertEqual(len({r["id"] for r in rows}), 40)
        self.assertEqual({s: sum(r["subset"] == s for r in rows) for s in {
            "routing", "sql", "retrieval", "deep"
        }}, {"routing": 10, "sql": 10, "retrieval": 10, "deep": 10})

    def test_sse_parser(self):
        raw = [
            b"event: state\n",
            b'data: {"task_type":"direct"}\n',
            b"\n",
            b"event: complete\n",
            b'data: {"status":"done"}\n',
            b"\n",
        ]
        events = list(self.runner.parse_sse(raw))
        self.assertEqual(events[0]["event"], "state")
        self.assertEqual(events[1]["data"]["status"], "done")

    def test_resume_fingerprint_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(
                json.dumps({"service_status": {"config_fingerprint": "abc"}}),
                encoding="utf-8",
            )
            old = json.loads(path.read_text(encoding="utf-8"))
            self.assertNotEqual(old["service_status"]["config_fingerprint"], "def")

    def test_trace_validation(self):
        good = [{
            "id": "x",
            "status": "done",
            "ablation_trace": {
                "variant": "wo_images",
                "injected_image_count": 0,
            },
        }]
        bad = [{
            "id": "x",
            "status": "done",
            "ablation_trace": {
                "variant": "wo_images",
                "injected_image_count": 1,
            },
        }]
        self.assertEqual(self.analyzer.validate_trace("wo_images", good), [])
        self.assertTrue(self.analyzer.validate_trace("wo_images", bad))


if __name__ == "__main__":
    unittest.main()
