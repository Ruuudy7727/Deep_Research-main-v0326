# -*- coding: utf-8 -*-
"""FunASR 本地语音识别（懒加载 + 单线程推理锁）。

环境变量见项目根目录 readme_ASR_funasr.md。
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, List, Optional

_funasr_lock = threading.Lock()
_model: Any = None
_load_error: Optional[str] = None


def funasr_installed() -> bool:
    return importlib.util.find_spec("funasr") is not None


def is_enabled() -> bool:
    if not os.getenv("FUNASR_ENABLED", "1").strip().lower() in {"1", "true", "yes", "y", "on"}:
        return False
    return funasr_installed()


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def model_loaded() -> bool:
    return _model is not None


def last_load_error() -> Optional[str]:
    return _load_error


def max_upload_bytes() -> int:
    try:
        mb = float(os.getenv("FUNASR_MAX_UPLOAD_MB", "32"))
    except ValueError:
        mb = 32.0
    return int(max(1, mb) * 1024 * 1024)


def _env_model() -> str:
    return os.getenv("FUNASR_MODEL", "paraformer-zh").strip() or "paraformer-zh"


def _env_device() -> str:
    return os.getenv("FUNASR_DEVICE", "cuda:0").strip() or "cuda:0"


def _env_hub() -> str:
    return os.getenv("FUNASR_HUB", "ms").strip() or "ms"


def _normalize_result(res: Any) -> str:
    if res is None:
        return ""
    if isinstance(res, str):
        return res.strip()
    if isinstance(res, dict):
        t = res.get("text") or res.get("pred") or res.get("value")
        if isinstance(t, str):
            return t.strip()
    if isinstance(res, list):
        parts: List[str] = []
        for item in res:
            if isinstance(item, dict):
                t = item.get("text") or item.get("pred")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        return " ".join(parts).strip() if parts else ""
    return str(res).strip()


def _load_model_unlocked() -> Any:
    global _model, _load_error
    if _model is not None:
        return _model
    _load_error = None
    try:
        from funasr import AutoModel  # type: ignore

        _model = AutoModel(
            model=_env_model(),
            device=_env_device(),
            hub=_env_hub(),
        )
        return _model
    except Exception as e:
        _load_error = f"{type(e).__name__}: {e}"
        raise


def convert_to_wav16k_mono(src_path: Path, suffix: str) -> Path:
    """将上传音频转为 16k 单声道 WAV。浏览器 webm/opus 依赖 ffmpeg。"""
    out_fd, out_str = tempfile.mkstemp(suffix=".wav")
    os.close(out_fd)
    out_path = Path(out_str)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [
            ffmpeg,
            "-nostdin",
            "-y",
            "-i",
            str(src_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "wav",
            str(out_path),
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=int(os.getenv("FUNASR_FFMPEG_TIMEOUT_SEC", "120")),
            )
        except subprocess.CalledProcessError as e:
            err = (e.stderr or e.stdout or "").strip()
            raise RuntimeError(f"ffmpeg 转码失败: {err[:800]}") from e
        return out_path

    if suffix.lower() == ".wav":
        return src_path
    raise RuntimeError("未检测到 ffmpeg：无法将 webm 等转为 WAV，请安装 ffmpeg 或上传 wav。")


def transcribe_file(src_path: Path, original_suffix: str) -> str:
    """对本地文件路径做识别，返回纯文本。"""
    if not is_enabled():
        raise RuntimeError("FunASR 未启用或未安装 funasr 包。")

    wav_path: Optional[Path] = None
    try:
        wav_path = convert_to_wav16k_mono(src_path, original_suffix)
        with _funasr_lock:
            model = _load_model_unlocked()
            res = model.generate(input=str(wav_path))
        return _normalize_result(res)
    finally:
        if wav_path is not None and wav_path != src_path and wav_path.exists():
            try:
                wav_path.unlink()
            except OSError:
                pass
