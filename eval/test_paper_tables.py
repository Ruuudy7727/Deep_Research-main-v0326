from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "eval" / "paper_tables.py"
    spec = importlib.util.spec_from_file_location("paper_tables", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class PaperTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paper = load_script()

    def test_result_metrics_are_traceable(self):
        rows = [
            {
                "id": "sql_1",
                "subset": "sql",
                "status": "done",
                "gold_action": "DATABASE",
                "observed_action": "DATABASE",
                "gold_tables": ["alarm_event"],
                "gold_sql_terms": ["pack-7"],
                "executed_sqls": ["select * from alarm_event where bmu_code='pack-7'"],
                "answer_elapsed_seconds": 10.0,
            },
            {
                "id": "ret_1",
                "subset": "retrieval",
                "status": "done",
                "gold_keywords": ["thermal runaway"],
                "evidence_chunks": [{"text": "thermal runaway mitigation"}],
                "answer_elapsed_seconds": 20.0,
            },
        ]
        metrics = self.paper.result_metrics(rows)
        self.assertEqual(metrics["task_count"], 2)
        self.assertEqual(metrics["sql_ready"], 1.0)
        self.assertEqual(metrics["retrieval_recall3"], 1.0)
        self.assertEqual(metrics["latency"], 15.0)

    def test_missing_annotations_render_tbd(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            text = self.paper.table_ii_main({"full": []}, {})
            self.assertIn("TBD", text)
            self.paper.write_docx("# Title\n\n## Table\n\n| A | B |", out / "x.docx")
            if (out / "x.docx").exists():
                self.assertGreater((out / "x.docx").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
