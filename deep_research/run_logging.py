# -*- coding: utf-8 -*-
"""Per-request run logging for web server (server_plus).

Creates ``log/runs/<run_id>/`` with structured ``events.jsonl``, ``main.log``,
and per-run ``db_chain.jsonl`` / ``db_plan_sanitizer.jsonl`` paths compatible
with ``research_agent_analyze.retrieve_battery_node``.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def get_log_profile() -> str:
    profile = str(os.getenv("LOG_PROFILE", "lite") or "lite").strip().lower()
    return profile if profile in {"lite", "full"} else "lite"


def apply_runtime_logging_defaults(profile: str) -> None:
    """Mirror thinkdepth_test: tune DB/SQL verbosity unless user already set."""
    if profile == "lite":
        os.environ.setdefault("DB_CONSOLE_VERBOSE", "0")
        os.environ.setdefault("SQL_PARSER_VERBOSE", "0")
    else:
        os.environ.setdefault("DB_CONSOLE_VERBOSE", "1")
        os.environ.setdefault("SQL_PARSER_VERBOSE", "1")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def web_run_logging_enabled() -> bool:
    raw = os.getenv("SERVER_PLUS_RUN_LOGGING", "1").strip().lower()
    return raw not in {"0", "false", "no", "off", "n"}


def maybe_create_web_run_session(
    project_root: Path,
    *,
    thread_id: str,
    mode: str,
) -> Optional["WebRunSession"]:
    if not web_run_logging_enabled():
        return None
    return WebRunSession(project_root, thread_id=thread_id, mode=mode)


class WebRunSession:
    """One Cobot chat turn: filesystem artifacts under ``log/runs/<run_id>/``."""

    def __init__(self, project_root: Path, *, thread_id: str, mode: str) -> None:
        self.project_root = Path(project_root)
        self.thread_id = thread_id
        self.mode = mode
        self.log_profile = get_log_profile()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"web_{stamp}_{secrets.token_hex(4)}"
        self.run_dir = self.project_root / "log" / "runs" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.main_log_path = self.run_dir / "main.log"
        self.events_path = self.run_dir / "events.jsonl"
        self.db_chain_log_path = self.run_dir / "db_chain.jsonl"
        self.db_plan_sanitizer_log_path = self.run_dir / "db_plan_sanitizer.jsonl"
        self.nodes_dir = self.run_dir / "nodes"

        if self.log_profile == "full":
            self.nodes_dir.mkdir(parents=True, exist_ok=True)

        self._write_main_header()

        self.event(
            "INFO",
            "run",
            "session_created",
            mode=self.mode,
            thread_id=self.thread_id,
            log_profile=self.log_profile,
            run_dir=str(self.run_dir),
            db_chain_log_path=str(self.db_chain_log_path),
        )

    def _write_main_header(self) -> None:
        lines = [
            f"run_id={self.run_id}",
            f"thread_id={self.thread_id}",
            f"mode={self.mode}",
            f"log_profile={self.log_profile}",
            f"run_dir={self.run_dir}",
            "---",
        ]
        try:
            self.main_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            pass

    def main_line(self, msg: str) -> None:
        try:
            with open(self.main_log_path, "a", encoding="utf-8") as f:
                f.write(msg.rstrip() + "\n")
        except OSError:
            pass

    def event(self, level: str, component: str, event: str, **extra: Any) -> None:
        row: Dict[str, Any] = {
            "ts": _utc_now_iso(),
            "level": level,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "mode": self.mode,
            "component": component,
            "event": event,
        }
        if extra:
            row["extra"] = extra
        try:
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    @staticmethod
    def preview_data(data: Any, *, max_keys: int = 48, max_str: int = 2000) -> Any:
        """Shrink arbitrary LangGraph event payloads for JSON logs."""

        def _walk(obj: Any, depth: int) -> Any:
            if depth <= 0:
                return "<max_depth>"
            if obj is None or isinstance(obj, (bool, int, float)):
                return obj
            if isinstance(obj, str):
                s = obj.replace("\n", "\\n")
                return s if len(s) <= max_str else s[:max_str] + f"...(+{len(s) - max_str} chars)"
            if isinstance(obj, dict):
                out: Dict[str, Any] = {}
                for i, (k, v) in enumerate(obj.items()):
                    if i >= max_keys:
                        out["_truncated_keys"] = len(obj) - max_keys
                        break
                    ks = str(k)
                    if ks == "output" and isinstance(v, dict):
                        out[ks] = {
                            "_type": "dict",
                            "_keys": list(v.keys())[:60],
                            "_len_keys": len(v.keys()),
                            "_approx_chars": len(json.dumps(v, default=str)),
                        }
                    elif ks == "input":
                        out[ks] = _walk(v, depth - 1)
                    else:
                        out[ks] = _walk(v, depth - 1)
                return out
            if isinstance(obj, (list, tuple)):
                lst = list(obj)
                prev = [_walk(x, depth - 1) for x in lst[:24]]
                if len(lst) > 24:
                    prev.append({"_truncated_items": len(lst) - 24})
                return prev
            return _walk(str(obj), depth - 1)

        try:
            return _walk(data, 5)
        except Exception as exc:  # noqa: BLE001
            return {"preview_error": str(exc)}

    def log_langgraph_full(
        self,
        kind: Optional[str],
        name: Optional[str],
        data: Any,
        graph_run_id: Any = None,
    ) -> None:
        if self.log_profile != "full":
            return
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name or "unnamed"))
        path = self.nodes_dir / f"node_{safe_name}.log"
        record = {
            "ts": _utc_now_iso(),
            "kind": kind,
            "name": name,
            "graph_run_id": graph_run_id,
            "data": self.preview_data(data),
        }
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass
