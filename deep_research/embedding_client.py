#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "text-embedding-v4"
_DEFAULT_TIMEOUT = 120.0
_MIDEA_HOST_MARKERS = ("aimpapi.midea.com", "midea.com")


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=str(_PROJECT_ROOT / ".env"), override=False)
    except Exception:
        pass


def _parse_optional_int(value: str) -> Optional[int]:
    text = (value or "").strip()
    if not text:
        return None
    return int(text)


def _is_midea_endpoint(base_url: str) -> bool:
    host = (base_url or "").lower()
    return any(marker in host for marker in _MIDEA_HOST_MARKERS)


def get_embedding_settings() -> Dict[str, object]:
    _load_env()
    base_url = (os.getenv("EMBED_BASE_URL", "").strip() or _DEFAULT_BASE_URL).rstrip("/")
    api_key = (
        os.getenv("EMBED_API_KEY", "").strip()
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
        or os.getenv("QWEN_API_KEY", "").strip()
    )
    model = os.getenv("EMBED_MODEL", "").strip() or _DEFAULT_MODEL
    # Default dimensions for text-embedding-v4; leave unset for other models unless configured.
    dimensions_raw = os.getenv("EMBED_DIMENSIONS", "").strip()
    if not dimensions_raw and model.lower().startswith("text-embedding-v4"):
        dimensions_raw = "1024"
    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "user": os.getenv("MIDEA_AIGC_USER", "").strip(),
        "dimensions": _parse_optional_int(dimensions_raw),
        "timeout": float(os.getenv("EMBED_TIMEOUT", str(_DEFAULT_TIMEOUT)).strip() or _DEFAULT_TIMEOUT),
        "is_midea": _is_midea_endpoint(base_url),
    }


class OnlineEmbeddingClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        user: str = "",
        dimensions: Optional[int] = None,
        timeout: float = _DEFAULT_TIMEOUT,
        require_midea_user: Optional[bool] = None,
    ) -> None:
        if not api_key:
            raise ValueError("EMBED_API_KEY 未设置，无法调用在线 embedding 服务。")
        if not base_url:
            raise ValueError("EMBED_BASE_URL 未设置，无法调用在线 embedding 服务。")
        if not model:
            raise ValueError("EMBED_MODEL 未设置，无法调用在线 embedding 服务。")

        is_midea = _is_midea_endpoint(base_url) if require_midea_user is None else bool(require_midea_user)
        if is_midea and not user:
            raise ValueError("MIDEA_AIGC_USER 未设置，无法调用美的在线 embedding 服务。")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions
        headers = {}
        if is_midea and user:
            headers["AIGC-USER"] = user
        self._client = OpenAI(
            api_key=api_key,
            base_url=self.base_url,
            default_headers=headers or None,
            timeout=timeout,
            max_retries=2,
        )

    def _embed(self, texts: List[str]) -> List[List[float]]:
        clean_texts = [text if isinstance(text, str) else str(text) for text in texts]
        request_args = {"model": self.model, "input": clean_texts}
        if self.dimensions is not None:
            request_args["dimensions"] = self.dimensions
        response = self._client.embeddings.create(**request_args)
        data = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(self, text: str) -> List[float]:
        vectors = self._embed([text])
        return vectors[0] if vectors else []


def build_embedding_client(
    *,
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
) -> OnlineEmbeddingClient:
    settings = get_embedding_settings()
    return OnlineEmbeddingClient(
        base_url=str(settings["base_url"]),
        api_key=str(settings["api_key"]),
        model=model or str(settings["model"]),
        user=str(settings["user"] or ""),
        dimensions=dimensions if dimensions is not None else settings["dimensions"],
        timeout=float(settings["timeout"]),
        require_midea_user=bool(settings.get("is_midea")),
    )


def embed_documents(texts: List[str], *, model: Optional[str] = None, dimensions: Optional[int] = None) -> List[List[float]]:
    client = build_embedding_client(model=model, dimensions=dimensions)
    return client.embed_documents(texts)


def embed_query(text: str, *, model: Optional[str] = None, dimensions: Optional[int] = None) -> List[float]:
    client = build_embedding_client(model=model, dimensions=dimensions)
    return client.embed_query(text)
