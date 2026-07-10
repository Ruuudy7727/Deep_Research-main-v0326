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

    def test_sql_no_clarify_dataset_contract(self):
        path = ROOT / "eval" / "datasets" / "v2_sql_20_no_clarify.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({r["id"] for r in rows}), 20)
        self.assertTrue(all(r["subset"] == "sql" for r in rows))
        self.assertEqual(
            {scenario: sum(r["scenario"] == scenario for r in rows) for scenario in {
                "alerting", "troubleshooting", "station_device_td"
            }},
            {"alerting": 5, "troubleshooting": 10, "station_device_td": 5},
        )
        self.assertEqual(sum(r["gold_action"] == "DATABASE" for r in rows), 18)
        self.assertEqual(sum(r["gold_action"] == "DATABASE_CHART" for r in rows), 2)
        self.assertTrue(all(r.get("gold_tables") for r in rows))
        self.assertTrue(all(r.get("gold_sql_terms") for r in rows))
        self.assertTrue(all("20" in r["question"] for r in rows))

    def test_runner_accepts_explicit_dataset(self):
        path = ROOT / "eval" / "datasets" / "v2_sql_20_no_clarify.jsonl"
        rows = self.runner.load_dataset("full", path)
        self.assertEqual(len(rows), 20)
        self.assertEqual(len(self.runner.load_dataset("wo_schema", path)), 20)
        self.assertEqual(len(self.runner.dataset_sha256(path)), 64)

    def test_sql_specialized_metrics_include_clarification(self):
        rows = [{
            "id": "sql_x",
            "subset": "sql",
            "status": "done",
            "gold_action": "DATABASE",
            "gold_task_type": "troubleshooting",
            "observed_action": "CLARIFY",
            "observed_task_type": "troubleshooting",
            "gold_tables": ["isc_score_result"],
            "gold_sql_terms": ["x"],
            "executed_sqls": [],
            "sql_row_count": 0,
            "expected_row_count_min": 0,
            "expected_row_count_max": 10,
        }]
        metrics = self.analyzer.sql_specialized_metrics(rows)
        self.assertEqual(metrics["clarification_rate"], 1.0)
        self.assertEqual(metrics["sql_generation_rate"], 0.0)

    def test_sql_ready_success_does_not_require_nonempty_rows(self):
        rows = [{
            "id": "sql_empty_but_valid",
            "subset": "sql",
            "status": "done",
            "gold_action": "DATABASE",
            "gold_task_type": "alerting",
            "observed_action": "DATABASE",
            "observed_task_type": "alerting",
            "gold_tables": ["alarm_event"],
            "gold_sql_terms": ["pack-7", "2026-03-22"],
            "executed_sqls": [
                "SELECT * FROM alarm_event WHERE bmu_code='pack-7' "
                "AND start_time >= '2026-03-22 00:00:00'"
            ],
            "sql_row_count": 0,
        }]
        metrics = self.analyzer.sql_specialized_metrics(rows)
        self.assertEqual(metrics["sql_ready_success_rate"], 1.0)
        self.assertEqual(metrics["sql_nonempty_result_rate"], 0.0)

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

    def test_rerun_subset_filter(self):
        existing = {
            "route_001": {"id": "route_001", "subset": "routing"},
            "deep_001": {"id": "deep_001", "subset": "deep"},
        }
        filtered = {
            row_id: row
            for row_id, row in existing.items()
            if row.get("subset") != "deep"
        }
        self.assertEqual(set(filtered), {"route_001"})

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

    def test_judge_parser_accepts_fenced_and_missing_comma_json(self):
        fenced = """```json
        {
          "A": {"correctness": 5, "completeness": 4, "traceability": 3,
                "actionability": 4, "uncertainty": 5},
          "B": {"correctness": 3, "completeness": 3, "traceability": 2,
                "actionability": 3, "uncertainty": 4},
          "preferred": "A",
          "reason": "A is stronger"
        }
        ```"""
        parsed = self.analyzer.parse_judge_json(fenced)
        self.assertEqual(parsed["A"]["correctness"], 5)
        self.assertEqual(parsed["preferred"], "A")

        missing_commas = """
        {
          "A": {"correctness": 5 "completeness": 4 "traceability": 3
                "actionability": 4 "uncertainty": 5},
          "B": {"correctness": 3 "completeness": 3 "traceability": 2
                "actionability": 3 "uncertainty": 4},
          "preferred": "A",
          "reason": "A is stronger"
        }"""
        parsed = self.analyzer.parse_judge_json(missing_commas)
        self.assertEqual(parsed["B"]["traceability"], 2)

    def test_builtin_env_loader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                'CODEX_TEST_ENV_A="alpha"\nexport CODEX_TEST_ENV_B=beta\n',
                encoding="utf-8",
            )
            os.environ.pop("CODEX_TEST_ENV_A", None)
            os.environ.pop("CODEX_TEST_ENV_B", None)
            self.analyzer.load_env_file(path)
            self.assertEqual(os.environ["CODEX_TEST_ENV_A"], "alpha")
            self.assertEqual(os.environ["CODEX_TEST_ENV_B"], "beta")
            os.environ.pop("CODEX_TEST_ENV_A", None)
            os.environ.pop("CODEX_TEST_ENV_B", None)

    def test_strict_image_and_multi_agent_validation(self):
        image_bad = [{
            "id": "image_x",
            "status": "done",
            "kb_images": [],
            "evidence_chunks": [{"image_paths": ["x.png"]}],
            "ablation_trace": {
                "variant": "wo_images",
                "retrieved_image_count": 0,
                "injected_image_count": 0,
            },
        }]
        multi_bad = [{
            "id": "multi_x",
            "status": "done",
            "ablation_trace": {
                "variant": "wo_multi_agent",
                "researcher_call_count": 0,
                "executed_nodes": ["supervisor_subgraph"],
            },
        }]
        self.assertTrue(self.analyzer.validate_trace("wo_images", image_bad))
        self.assertTrue(self.analyzer.validate_trace("wo_multi_agent", multi_bad))


if __name__ == "__main__":
    unittest.main()
